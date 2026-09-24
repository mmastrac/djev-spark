#!/usr/bin/env bash
# Boot vLLM with the perf overlay, then the structured server in front of it.
# Refuses to start without memory headroom: on unified memory a CUDA
# overshoot is a host OOM and a hang, not a failed request.
set -euo pipefail

MODEL=${MODEL:-/models/dgemma}
SERVED_NAME=${SERVED_NAME:-dgemma}
# Served canvas, and the block width generation uses by load (adaptive-canvas):
# a lightly loaded server denoises wide blocks, a busy one narrow blocks, so
# 256 wins single-stream and 64 at 32 concurrent. Reads set their own width.
CANVAS=${CANVAS:-256}
CANVAS_SCHEDULE=${CANVAS_SCHEDULE-[[1, 2, 256], [3, 6, 128], [7, 32, 64]]}
MAX_SEQS=${MAX_SEQS:-32}               # 128 halved throughput (graph coverage)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.40}
ATTN=${ATTN:-TRITON_ATTN}
PORT=${PORT:-8010}
STRUCTURED_PORT=${STRUCTURED_PORT:-8011}
TLS_PORT=${TLS_PORT:-0}
KV_CACHE_GB=${KV_CACHE_GB:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
HEADROOM_GB=${HEADROOM_GB:-12}
# Reads over the labels only (constrained-vocab), and a fixed sample count as
# one request (diffusion-samples). 0 turns either off in the server.
CONSTRAINED=${CONSTRAINED:-1}
ENGINE_SAMPLES=${ENGINE_SAMPLES:-1}
MAX_SAMPLES=${MAX_SAMPLES:-32}
# The model writes Gemma's thought channel and still writes an EMPTY one with
# thinking off; without the parser every plain chat answer starts "thought\n".
REASONING_PARSER=${REASONING_PARSER-gemma4}
# Tool calls in every tool_choice mode need a parser. The server pins the
# call's opening for "required" and a named function, which vLLM cannot
# enforce on a diffusion model. Empty disables tool calling.
TOOL_CALL_PARSER=${TOOL_CALL_PARSER-gemma4}
TEST_PAGE=${TEST_PAGE:-}
WAIT_SECS=${WAIT_SECS:-1800}
EXTRA_ARGS=${EXTRA_ARGS:-}

[[ -f "$MODEL/config.json" ]] || { echo "no model at $MODEL; run scripts/download-model.sh" >&2; exit 2; }

# Start-up runs the sampler at the full batch. The fused sampler's one-pass
# row statistics replace most of the fp32 [rows, vocab] copies the old
# sampler made (ten); TRANSIENT_COPIES is how many such copies to budget for.
# Measured at 32 x 256: the whole container peaked at 33 GB, weights and KV
# included, so about 1.6 copies.
TRANSIENT_COPIES=${TRANSIENT_COPIES:-2}
WEIGHTS_GB=19
TRANSIENT_GB=$(( MAX_SEQS * CANVAS * 262144 * 4 * TRANSIENT_COPIES / 1073741824 + 1 ))
NEED_GB=$(( WEIGHTS_GB + KV_CACHE_GB + TRANSIENT_GB + HEADROOM_GB ))
AVAIL_GB=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
echo "memory: ${AVAIL_GB} GB available, need ${NEED_GB} (weights ${WEIGHTS_GB} + KV ${KV_CACHE_GB} + start-up transient ${TRANSIENT_GB} + headroom ${HEADROOM_GB})"
if (( AVAIL_GB < NEED_GB )); then
  echo "refusing to start: not enough memory; stop other models first" >&2
  exit 2
fi

DIFFUSION_CONFIG="{\"canvas_length\": ${CANVAS}, \"max_samples\": ${MAX_SAMPLES}"
[[ -n "$CANVAS_SCHEDULE" ]] && DIFFUSION_CONFIG+=", \"canvas_length_per_batch_size\": ${CANVAS_SCHEDULE}"
DIFFUSION_CONFIG+="}"

healthy() {
  python3 - "$1" <<'EOF'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2)
except Exception:
    sys.exit(1)
EOF
}

# --async-scheduling: per-request canvas widths, which every read uses, and
# the adaptive schedule need it. --language-model-only: text serving; the
# multimodal-prefix path narrows the backend choice.
# shellcheck disable=SC2086  # EXTRA_ARGS is a flag list
vllm serve "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code \
  --language-model-only --async-scheduling \
  --max-num-seqs "$MAX_SEQS" --max-model-len "$MAX_MODEL_LEN" \
  --attention-backend "$ATTN" --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-memory $(( KV_CACHE_GB * 1073741824 )) \
  ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
  --max-logprobs 128 --enable-prefix-caching \
  ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
  ${TOOL_CALL_PARSER:+--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER" --exclude-tools-when-tool-choice-none} \
  --diffusion-config "$DIFFUSION_CONFIG" \
  --override-generation-config '{"max_new_tokens": null}' \
  --port "$PORT" $EXTRA_ARGS &
VLLM_PID=$!

for (( i = 0; i < WAIT_SECS / 5; i++ )); do
  healthy "$PORT" && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm exited during start-up" >&2; exit 1; }
  sleep 5
done
healthy "$PORT" || { echo "vllm not healthy after ${WAIT_SECS}s" >&2; kill "$VLLM_PID"; exit 1; }
echo "vllm ready on :${PORT}"

SERVER_ARGS=(--upstream "http://127.0.0.1:${PORT}" --model "$SERVED_NAME" --tokenizer "$MODEL"
  --canvas "$CANVAS" --port "$STRUCTURED_PORT" --tls-port "$TLS_PORT" --cert-dir /root/.cache/djev
  --max-samples "$MAX_SAMPLES")
[[ "$CONSTRAINED" == 1 ]] && SERVER_ARGS+=(--constrained)
[[ "$ENGINE_SAMPLES" == 1 ]] && SERVER_ARGS+=(--engine-samples)
[[ "$TEST_PAGE" == 1 ]] && SERVER_ARGS+=(--pages /opt/dgemma/pages)
echo "structured server: ${SERVER_ARGS[*]}"

# The structured server is restarted whenever it exits, so its code can be
# reloaded (docker cp the file in, then pkill -f structured_server.py)
# without touching vLLM. Only vLLM's exit ends the container.
serve_structured() {
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    # A reload signals the server, and set -e would take this loop down with it.
    python3 /opt/dgemma/structured_server.py "${SERVER_ARGS[@]}" || true
    echo "structured server exited; restarting" >&2
    sleep 1
  done
}
serve_structured &
SERVER_LOOP=$!

trap 'kill "$VLLM_PID" "$SERVER_LOOP" 2>/dev/null; pkill -f structured_server.py 2>/dev/null; wait' TERM INT
wait "$VLLM_PID"
echo "vllm exited; stopping" >&2
kill "$SERVER_LOOP" 2>/dev/null
pkill -f structured_server.py 2>/dev/null
wait
exit 1
