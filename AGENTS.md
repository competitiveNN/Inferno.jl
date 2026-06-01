# Inferno.jl Development Guide

## Current Status: GGUF + Safetensors CPU Inference Working

Qwen3.5-0.8B-VL CPU inference produces coherent multi-token text and matches the HF reference output.

Verified prompt/output examples are in `HISTORY.md`.

Performance: per-token allocations ~10KB; throughput 14-18 tokens/sec on the current CPU backend.

## Active Rules

- Never claim inference success from a single-token test; verify at least 64-128 tokens of coherent generated text.
- Treat the HuggingFace transformers model implementation as the ground truth for architecture details.
- Prefer profiling-driven optimization, and record concrete metrics.

## Model Reference Pointers

HuggingFace reference implementations to check when implementing or debugging model support:

- `~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/tokenization_qwen3_5.py`

## Debugging Shortcuts

Use DaemonMode.jl to avoid REPL startup cost during iterative test runs.

## Known Active Constraints

- Chat templates for hybrid thinking models must include thinking tokens; otherwise output becomes incoherent.
- Generation should stop on EOS/end-of-turn tokens; otherwise the model continues into extra turns.

For older phase summaries, benchmark details, and completed-work history, see `HISTORY.md`.
