#!/usr/bin/env python3
"""Assert the base, the perf overlay and the server still fit together.

Failing the build is much cheaper than an image that boots, loads 18 GB of
weights, and then serves reads slowly or wrong. Each check names the branch
in vllm-stack.tsv (or the djev PR) whose change it looks for.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import torch
from vllm.model_executor.models.registry import ModelRegistry

problems = []


def source(module):
    spec = importlib.util.find_spec(module)
    return Path(spec.origin).read_text() if spec and spec.origin else ""


if "DiffusionGemmaForBlockDiffusion" not in set(ModelRegistry.get_supported_archs()):
    problems.append("DiffusionGemmaForBlockDiffusion is not in this vLLM's registry")

model = source("vllm.model_executor.models.diffusion_gemma")
for needle, what in (
    ("diffusion_pinned", "structured reads (vllm-project/vllm#57250)"),
    ("diffusion_read_only", "structured reads (vllm-project/vllm#57250)"),
    ("diffusion_seed_canvas", "structured reads (vllm-project/vllm#57250)"),
    ("recompile_limit=64", "the sampler's dynamo recompile limit"),
    ("step_allowed", "constrained-vocab"),
    ("sc_skip", "sc-last-step-skip"),
    ("sample_row_stats", "fused-sampler"),
    ("renoise_slots", "diffusion-samples"),
    ("canvas_full", "adaptive-canvas"),
):
    if needle not in model:
        problems.append(f"diffusion_gemma.py lacks {needle}: {what} did not take")

for module, needle, what in (
    ("vllm.config.diffusion", "canvas_length_per_batch_size", "adaptive-canvas"),
    ("vllm.config.diffusion", "max_samples", "diffusion-samples"),
    ("vllm.v1.core.sched.diffusion_scheduler", "canvas_length_per_batch_size", "adaptive-canvas"),
    ("vllm.utils.diffusion", "diffusion_constrained", "constrained-vocab"),
    ("vllm.utils.diffusion", "diffusion_samples", "diffusion-samples"),
    ("vllm.v1.attention.backends.flashinfer", "_plan_split_prefill", "flashinfer-per-request-causal"),
    ("vllm.model_executor.models.diffusion_gemma_sampler", "safe_m", "fused-sampler-masked-block"),
):
    if needle not in source(module):
        problems.append(f"{module} lacks {needle}: {what} did not take")

# Reading the files is not enough: import them, so a bad merge fails the
# build instead of the model-inspection subprocess at boot.
for module in (
    "vllm.model_executor.models.diffusion_gemma",
    "vllm.model_executor.models.diffusion_gemma_sampler",
    "vllm.v1.core.sched.diffusion_scheduler",
    "vllm.v1.attention.backends.flashinfer",
):
    try:
        importlib.import_module(module)
    except Exception as e:
        problems.append(f"{module} does not import: {e!r}")
try:
    from vllm.config.diffusion import DiffusionConfig

    DiffusionConfig(
        canvas_length=256,
        canvas_length_per_batch_size=[(1, 2, 256), (3, 6, 128), (7, 32, 64)],
        max_samples=32,
    )
except Exception as e:
    problems.append(f"DiffusionConfig rejects the entrypoint's schedule: {e!r}")

server = Path("/opt/dgemma/structured_server.py").read_text()
for needle, what in (
    ("def _models", "GET /v1/models (mmastrac/djev#5)"),
    ("def _relay", "ordinary chat passed through (mmastrac/djev#6)"),
    ("def span_text_of", "spans grounded in an object's text (mmastrac/djev#8)"),
    ("def pin_tool_choice", "required and named tool_choice (mmastrac/djev#9)"),
    ('"--constrained"', "the --constrained flag the entrypoint passes"),
    ('"--engine-samples"', "the --engine-samples flag the entrypoint passes"),
    ('"--pages"', "the --pages flag the entrypoint passes"),
):
    if needle not in server:
        problems.append(f"structured server lacks {what}")
for page in ("index", "walk", "cube"):
    if not Path(f"/opt/dgemma/pages/{page}.html").is_file():
        problems.append(f"pages/{page}.html is missing")

try:
    from vllm.reasoning import ReasoningParserManager

    ReasoningParserManager.get_reasoning_parser("gemma4")
except Exception as e:
    problems.append(f"no gemma4 reasoning parser; plain chat would lead with 'thought': {e!r}")
try:
    from vllm.tool_parsers import ToolParserManager

    ToolParserManager.get_tool_parser("gemma4")
except Exception as e:
    problems.append(f"no gemma4 tool parser; every tool_choice but none would 400: {e!r}")

if "sm_120" not in torch.cuda.get_arch_list():
    problems.append(f"no sm_120 cubins in torch {torch.__version__}")

if problems:
    for p in problems:
        print(f"IMAGE CHECK FAILED: {p}", file=sys.stderr)
    sys.exit(1)
print("image ok")
