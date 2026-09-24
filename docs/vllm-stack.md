# The vLLM perf stack

Structured reads are upstream ([vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250),
merged 2026-09-22), so a stock nightly serves this image's reads. The image
adds seven perf branches from [mmastrac/vllm](https://github.com/mmastrac/vllm)
on top. [`vllm-stack.tsv`](../vllm-stack.tsv) lists them with the fork head each
one is pinned at. The Dockerfile's `VLLM_REF` is the fork branch
`djev-spark-stack`: the nightly's own commit plus each branch's feature diff.

| branch | PR | what it adds |
|---|---|---|
| `constrained-vocab` | [58216](https://github.com/vllm-project/vllm/pull/58216) | reads over `logprob_token_ids` only (`diffusion_constrained`). The unembedding, sampler and self-conditioning run over the K labels instead of the whole vocabulary: the same argmax for about a quarter less GPU time per read |
| `sc-last-step-skip` | [58221](https://github.com/vllm-project/vllm/pull/58221) | skips the self-conditioning matmul on a read's last step, whose result no step would use |
| `fused-sampler` | [58226](https://github.com/vllm-project/vllm/pull/58226) | one Triton pass per row for argmax, Gumbel sample, entropy and softmax, in `diffusion_gemma_sampler.py`. Its last commit keeps a row finite when `top_k` / `top_p` mask a whole leading block: the running max stayed `-inf`, and `-inf - -inf` made the rest of the row NaN |
| `diffusion-entropy-masked` | [58440](https://github.com/vllm-project/vllm/pull/58440) | finite entropy when `top_k` / `top_p` leave `-inf` logits. Without it every block runs to its step cap, 4.4x slower |
| `diffusion-samples` | [58438](https://github.com/vllm-project/vllm/pull/58438) | `diffusion_samples: k` fans one seeded canvas into k noise draws in one request, seedable, capped by `diffusion_config.max_samples` |
| `flashinfer-per-request-causal` | none yet | FlashInfer takes the per-request causal tensor. It overlaps upstream PR 58015's `supports_mixed_causal()` gate |
| `adaptive-canvas` | none yet | `canvas_length_per_batch_size`: the async scheduler picks each generation block's width from the load |

The server side of two of them is in djev: `--constrained` and
`--engine-samples`, which the entrypoint turns on (`CONSTRAINED`,
`ENGINE_SAMPLES`).

## Why the stack is diffs, not merges

Some branches have upstream main merged in past the nightly (`fused-sampler`
sits on a main 37 commits newer than `e9757321`). Merging them would carry
those upstream commits into the overlay, on top of an image built without
them. The stack instead applies each branch's diff against its own merge
base with upstream main, so the overlay is exactly the manifest's branches. Two
overlaid files had changed upstream in that window (`vllm/v1/attention/backend.py`
and a scheduler test); both take the nightly's version plus the branch diff.

The Dockerfile's fork stage fails the build if `VLLM_REF` does not sit on
`VLLM_BASE`, or if any head in `vllm-stack.tsv` is not named in the stack's
history. `patches/verify_image.py` then checks that each branch's change is
in the installed vLLM, that the changed modules import, and that the server
has the flags the entrypoint passes.

## Keeping it current

```bash
scripts/vllm-stack.sh status
```

prints one line per branch: whether the fork branch moved past its pinned
head, the upstream PR's state, and a verdict.

| verdict | what to do |
|---|---|
| `ok` | nothing |
| `moved; rebuild the stack` | update the head in `vllm-stack.tsv` and rebuild |
| `merged upstream; drop once BASE moves past it` | wait for a nightly that contains the merge |
| `drop: merged and in the base nightly` | delete the line and rebuild; the nightly has it |

To rebuild, on a nightly that has been published for the platform
(`docker manifest inspect vllm/vllm-openai:nightly-<commit>` shows arm64):

```bash
scripts/vllm-stack.sh build <nightly commit> ./.vllm-stack
# resolve each conflict as below, git add, commit with the message it prints, rerun
git -C .vllm-stack push origin djev-spark-stack-new:djev-spark-stack-<nightly short>
```

then set `BASE`, `VLLM_BASE` (both places) and `VLLM_REF` in the Dockerfile.

### Known conflicts

All in `diffusion_gemma.py` or its test unless noted. They resolve the same
way each time.

- `sc-last-step-skip` after `constrained-vocab`: adjacent additions to the
  request states (`constrained` / `step_allowed` beside `max_steps_np` /
  `denoise_steps_np`) and to the sampler step. Keep both; compute `sc_skip`
  before the constrained view of the logits.
- `fused-sampler` after `constrained-vocab`: the fused branch moves sampling
  out of `_compiled_sample_step`, which no longer takes `allowed`. Take the
  fused side, and map the K-space picks back to token ids in the caller,
  right after `sample_row_stats`: `argmax_rows = allowed[argmax_rows]`,
  `sample_rows = allowed[sample_rows]`. Drop `allowed=allowed` from the call.
  The test imports keep both sides.
- `diffusion-entropy-masked` after `fused-sampler`: the fused kernel and its
  reference already drop `-inf` columns from the entropy sum, so take the
  fused side of the model. In the test, keep both tests and pass the
  helper's `logits` to `sample_row_stats_reference`.
- `diffusion-samples` after `constrained-vocab`, in the example server:
  `read_xargs` must include `**constrained_xargs()`. The README table keeps
  both rows.
- `adaptive-canvas` after `fused-sampler` and `diffusion-samples`:
  `DiffusionConfig` keeps both fields. The confidence phase takes the fused
  side plus adaptive-canvas's real-width mean
  (`real = arange(CL) < valid_canvas_len`). The test helper gains `valid`,
  and both sets of tests stay.

After any rebuild, grep for conflict markers before committing, and run the
tests that cover the stack inside the image:

```bash
pytest tests/v1/sample/test_diffusion_gemma_reads.py tests/v1/sample/test_diffusion_sampler_stats.py
pytest tests/test_sampling_params.py tests/config/test_model_arch_config.py -k "diffusion or gemma4"
pytest tests/v1/core/test_scheduler.py -k "diffusion or canvas"
```

Two failures in the scheduler file are environmental (PP=3, and ngram on the
V2 runner).
