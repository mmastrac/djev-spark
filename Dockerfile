# syntax=docker/dockerfile:1
# DiffusionGemma NVFP4 structured reads on a DGX Spark (GB10, aarch64, CUDA 13).
#
# Nothing is compiled. The base is one of vLLM's per-commit nightly images,
# which are multi-arch and CUDA 13. Structured reads are upstream
# (vllm-project/vllm#57250), so the base serves them as is. On top goes the
# perf stack: the fork branch djev-spark-stack, which is the nightly's own
# commit plus the feature diffs of the branches in vllm-stack.tsv, python
# files only. The build copies the stack's changed vllm/ files over the
# base's site-packages.

ARG BASE=vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19

# --- the perf stack, at a pinned commit ---------------------------------------
FROM alpine/git:latest AS fork
ARG VLLM_FORK=https://github.com/mmastrac/vllm.git
ARG VLLM_REF=2751ece02505565231a1368b21f43825a1637192
ARG VLLM_BASE=e9757321527ca1ecd514c07c1418dd2c53da3d19
COPY vllm-stack.tsv /vllm-stack.tsv
# The stack must sit directly on the image's commit, so the overlay carries
# the perf branches and no other upstream change. Every head in
# vllm-stack.tsv must be named in the stack's history, so the manifest and
# VLLM_REF cannot drift apart.
RUN git clone --filter=blob:none --quiet "${VLLM_FORK}" /fork \
    && cd /fork \
    && git checkout --quiet "${VLLM_REF}" \
    && git merge-base --is-ancestor "${VLLM_BASE}" HEAD \
       || { echo "VLLM_REF is not built on VLLM_BASE ${VLLM_BASE}" >&2; exit 1; } \
    && log=$(git log --format=%B "${VLLM_BASE}..HEAD") \
    && grep -v '^#' /vllm-stack.tsv | while IFS="$(printf '\t')" read -r branch head pr parent; do \
         echo "$log" | grep -q "${head:0:10}" \
           || { echo "vllm-stack.tsv: ${branch} ${head:0:10} is not in the stack at ${VLLM_REF}" >&2; exit 1; }; \
         echo "stack: ${branch} ${head:0:10} (PR ${pr})"; \
       done \
    && git diff --name-only "${VLLM_BASE}" HEAD -- vllm > /fork/changed.txt \
    && git diff --name-only --diff-filter=A "${VLLM_BASE}" HEAD -- vllm > /fork/added.txt \
    && cat /fork/changed.txt

# --- the structured server -----------------------------------------------------
# github.com/mmastrac/djev is the canonical server; this image takes it at a
# pinned commit.
FROM alpine/git:latest AS djev
ARG DJEV_REPO=https://github.com/mmastrac/djev.git
ARG DJEV_REF=9a67e54
RUN git clone --quiet "${DJEV_REPO}" /djev \
    && cd /djev \
    && git checkout --quiet "${DJEV_REF}" \
    && git log -1 --format="djev server at %h: %s"

# --- the image -----------------------------------------------------------------
FROM ${BASE}
ARG VLLM_BASE=e9757321527ca1ecd514c07c1418dd2c53da3d19

# The base ships CUDA libraries without their headers and without some
# unversioned .so symlinks. FlashInfer's JIT needs both; see the script.
COPY patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# The perf overlay. The build stops if the base's vLLM is not the commit the
# stack was built on, or a target file is missing.
COPY patches/overlay_vllm.py /tmp/overlay_vllm.py
RUN --mount=type=bind,from=fork,source=/fork,target=/fork \
    python3 /tmp/overlay_vllm.py /fork "${VLLM_BASE}" && rm /tmp/overlay_vllm.py

# Worker memory cap, armed by TORCH_MEM_FRACTION at run time and inert
# otherwise. On unified memory an unbounded worker takes the host down rather
# than its own request.
COPY patches/spark_mem_trace.py /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
COPY patches/worker_memory_cap.py /tmp/worker_memory_cap.py
RUN python3 /tmp/worker_memory_cap.py && rm /tmp/worker_memory_cap.py

COPY --from=djev /djev/structured_server.py /opt/dgemma/structured_server.py
COPY pages/ /opt/dgemma/pages/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && python3 -m py_compile /opt/dgemma/structured_server.py

# Fails the build if the overlay did not take, the model no longer imports,
# or the server lacks a flag the entrypoint passes.
COPY patches/verify_image.py /tmp/verify_image.py
RUN python3 /tmp/verify_image.py && rm /tmp/verify_image.py

# 8010 vLLM (OpenAI API), 8011 structured decisions
EXPOSE 8010 8011
ENTRYPOINT ["/entrypoint.sh"]
