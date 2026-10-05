# Inferno.jl — Development Guide

Pure-Julia port of the **ninfer** inference engine (https://github.com/Neroued/ninfer)
to this local setup: 2x Intel Arc Pro B580 + Arrow Lake CPU.

> **Lineage of this file (2026-10-05).** This guide is a *translation* of the C++
> engine's `AGENTS.md` into the Julia tree. The C++ original lives in the sibling
> project `/var/home/fra/dev/intfer` (SYCL/Level-Zero, hand-written kernels, 27B
> TP=2); its doc is the source of every C++-era measurement referenced below. The
> copy that previously sat in this repo was a stale snapshot of intfer's revision
> `552dabc` — nothing unique was lost by replacing it. Translation policy: sections
> with a Julia counterpart are rewritten against this tree; sections that only
> describe the C++ kernel stack are **pointed at, not copied**; historical records
> are never reworded into agreement (see "History and audit trail").

## READ FIRST — what is live

* **CPU backend: working.** Qwen3.5-0.8B loads from safetensors
  (`/run/host/var/home/fra/data/models/safetensors/Qwen3.5-0.8B`) and GGUF
  (`INFERNO_MODEL_PATH`). Coherent multi-token generation verified against the
  HuggingFace reference; end-to-end pipeline check passes.
* **GPU backend: BLOCKED.** `using oneAPI` is wired, but GPU compute does not run
  on this box (`src/GPUInit.jl`): the system `intel-compute-runtime` v26.18
  GMM-aborts on `zeInit` for Battlemage (Xe2), and the bundled `NEO_jll` v25.44
  is too old for the Xe2 ISA. Element-wise KernelAbstractions kernels were
  verified numerically correct; a 24-layer GPU forward diverged on matmul
  precision (May 2026). CPU is the reference backend until the driver/NEO moves.
* **Julia 1.13.0**, Arrow Lake 20 cores, BLAS threads = 8 (measured optimal).
* **Test suite: `Pkg.test()` passes** — 8 live unit suites plus model-gated
  diagnostics when the models are present. See "Verification".
* **Box: both B580s enumerate** — but this has dropped and come back repeatedly
  with nothing in any repo changing. It is HOST-SIDE driver state, not a code
  fault. The recovery kit (`gpu_recover.sh diag` -> `rebind` -> verify) lives in
  `/var/home/fra/dev/intfer/scripts/` and operates on the host driver, so it
  applies from either project. **Never report "no GPU visible" without first
  checking the oneAPI environment** — a missing `setvars.sh`/PATH is a statement
  about the shell, not the machine. Never lead with the small-BAR theory: a plain
  rebind fixed it last time with no BAR change.

### Operator standing rules (carried over from the C++ program)

1. **Full-model long runs are the operator's to measure.** Multi-minute prefill /
   decode / 27B-class runs must not be launched from an agent session. Agents run
   the unit suites and small probes, and state which half of a claim is proven
   and which is pending the box.
2. **Do not commit while a full test run is in flight** — a second session
   committing mid-run makes the recorded HEAD name a different tree than the one
   tested.

## Architecture (src/)

`src/Inferno.jl:1` is the module root; include order there is the dependency
order. Map (roles, not exhaustive):

| file | role |
|---|---|
| `QuantsData.jl`, `Dequant.jl`, `QuantsCPU.jl`, `QuantMV.jl` | quantized formats (Q4_K/Q5_K/Q6_K/Q8_0...) and CPU matvec |
| `QuantizedKernels.jl` | native C SIMD kernels (`src/kernels/quant_kernels.c`, `bf16_avx2.c` -> prebuilt `.so`); disabled by `INFERNO_NO_QUANT_KERNELS` |
| `GGUF.jl`, `Safetensors.jl` | weight format parsers |
| `Model.jl`, `Qwen3.jl`, `Qwen35.jl`, `Gemma4*.jl`, `Jamba.jl` | model configs/structs (Qwen3.5 = hybrid SSM + full attention; Gemma4; Jamba) |
| `ModelCPU.jl` | CPU forward: SSM, GatedDeltaNet, full attention, MLP, RMSNorm, RoPE; exports `generate_*`, `GenerationState`, `create_generation_state`, `generate_batch` (`src/ModelCPU.jl:54`) |
| `Loader.jl`, `LoaderCPU.jl`, `*GPULoader.jl` | model loading (CPU path is the working one) |
| `GPUCommon.jl`, `GPUInit.jl`, `Qwen35GPU.jl`, `*GPU.jl` | oneAPI/GPU paths (blocked — see above) |
| `CommonOps.jl` | scratch-based ops incl. zero-allocation sampler (`SoftmaxScratch` + binary max-heap top-k) |
| `Engine.jl`, `Server.jl` | HTTP server (`start_server`; auth via `INFERNO_API_KEY`) |
| `Generate.jl` | high-level API: `generate_text`, `chat`, `SimpleTokenizer` (`src/Generate.jl:12`) |
| `ArrowLake.jl`, `AMXBF16.jl`, `BF16Support.jl` | CPU feature detection and (research-only) BF16 pipeline |

Entry points: `examples/basic_usage.jl`, `examples/inference.jl` (the main
playground), `examples/chat.jl`, `bin/chat.jl`.

## Generation correctness invariants

These are the port's hard-won gotchas; every one cost a debugging session
(see `HISTORY.md` and the memory notes):

* **Positions are 0-based internally.** Prefill starts at 0; the first sampled
  token is forwarded at `prompt_length` **before** the position increments.
  `curr_pos` must not increment before the first generated token is cached.
* **Every generated token must be cached before it becomes the next seed.**
  Single-token tests passing while multi-token output is garbage is the classic
  symptom of breaking this.
* **`BPETokenizer` IDs are 1-indexed**; stop-token helpers must use 1-indexed
  IDs (the chat stop helper once subtracted 1).
* **`reset_state!` resets SSM state and tracking, not cache contents** — the
  forward overwrites caches; do not "clear" them separately.
* **Stop tokens are checked immediately after sampling** and excluded from
  generated counts/output; `token_counts` includes prompt tokens; `max_tokens=0`
  and empty prompts must return cleanly.
* **Architecture conventions (Qwen3.5):** attention norms use the layernorm1p
  convention (+1) but SSM norms do not; Q/K use **L2 norm**, not RMSNorm;
  partial rotary on 25% of head_dim; decay is
  `exp(ssm_a * softplus(a + dt_bias))`; GGUF Conv1D stores `(C,K)` where the
  kernel expects `(K,C)`; safetensors BF16 is `UInt16 -> UInt32<<16 -> Float32`.
* **Chat templates for hybrid thinking models must include the thinking tokens**,
  and generation must stop on EOS/end-of-turn — without either, output goes
  incoherent or runs into extra turns.
* **Sampling edge cases:** empty logits, all-`-Inf`/NaN, and `temperature <= 0`
  (argmax fallback) are covered by `tests/unit/test_common_ops.jl`; very small
  temperatures need a numerically stable `exp`.

## Active rules

1. **Never claim inference success from a single token.** Verify at least 64-128
   tokens of coherent text; record the prompt and output.
2. **The HuggingFace `transformers` implementation is ground truth** for
   architecture details (`~/.local/lib/python*/site-packages/transformers/models/qwen3_5/`).
3. **A structural argument is not measurement.** The C++ program once diffed two
   kernel bodies statement-by-statement, found them identical, shipped — and
   decode went 180x slower. Only a real forward run catches it. Read the code,
   then run it.
4. **A fusion that changes FP32 accumulation order is not bit-exact** and needs a
   tolerance-based numerics contract agreed *before* code. Integer folds and
   exhaustively-verified converts are the safe fusions.
5. **Prefer profiling-driven optimization and record concrete metrics** — in
   `PERFORMANCE.md` / `HISTORY.md`, with the command and environment that
   produced them.

## Verification

```bash
julia --project=. -e 'using Pkg; Pkg.test()'     # full suite
```

`tests/runtests.jl` includes the 8 live unit suites from `tests/unit/`
(`test_gguf`, `test_tokenizer`, `test_engine`, `test_server_auth`,
`test_inferno_utils`, `core_components`, `test_common_ops`,
`test_generation_edges`) and, when the models are present, the model-gated
diagnostics `tests/diagnostics/check_generation_pipeline.jl` (safetensors) and
`check_bfloat16.jl` (GGUF). `tests/unit/` holds 43 files; only the included 8
are run by `Pkg.test` — the rest are standalone scripts (run manually, e.g.
`julia --project tests/unit/test_multi_token.jl`), as are `tests/legacy/` and
`tests/benchmark/`. `test/` is a symlink shim
(`test/runtests.jl -> ../tests/runtests.jl`, `test/unit -> ../tests/unit`) kept
for path compatibility.

Model paths are env-driven: `INFERNO_MODEL_PATH` (GGUF; default
`tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`) and
`INFERNO_SAFETENSORS_MODEL` (default
`/run/host/var/home/fra/data/models/safetensors/Qwen3.5-0.8B`).

## Performance

* **Current baseline** (`PERFORMANCE.md`): 14-19 tok/s on CPU for Qwen3.5-0.8B;
  per-token allocations ~10KB in the optimized forward path (was 2.7MB);
  persistent-KV API (`generate_with_cache`) cuts allocations ~60%.
* **Inherited program targets** (from the C++ program, 27B @ 2x B580):
  **40 tok/s** per-stream decode, **4000 tok/s** prefill. For this port the
  live objectives are CPU parity and unblocking the GPU path.
* **Measurement discipline** (the C++ program retired wall-time benchmarking only
  because its box could not hold `load1 < 2`; here wall-time *is* the instrument,
  with the same honesty rules):
  * pin `OMP_NUM_THREADS`/BLAS threads, exclude warmup, report **medians of >= 3
    runs with the spread**; a delta must clear the run-to-run spread, not merely
    be non-zero;
  * report allocation counts and GC pressure alongside wall time — allocation
    reduction has been the largest CPU win so far;
  * **never write a number you did not measure**; leave a blank `RESULT:` line
    for the operator's quiet-box run (`load1 < 2` for the whole arm).

## Environment variables — DERIVED, do not hand-maintain

```bash
grep -rhoE 'ENV\["[A-Za-z0-9_]+"' src/ tests/ bin/ examples/ | sort -u
```

| var | effect |
|---|---|
| `INFERNO_MODEL_PATH` | GGUF model path for tests/examples |
| `INFERNO_SAFETENSORS_MODEL` | safetensors dir for the pipeline diagnostic |
| `INFERNO_API_KEY` | HTTP server auth token (`Server.jl`) |
| `INFERNO_NO_QUANT_KERNELS` | disables the native C SIMD quant kernels (`src/QuantizedKernels.jl:28`; any value != `"0"` disables) |
| `INFERNO_KEEP_QUANTIZED` | opt-in to keeping weights quantized in-memory (memory savings; off by default — BLAS F32 is faster; set to `"1"` to enable, e.g. `INFERNO_KEEP_QUANTIZED=1`) |
| `OMP_NUM_THREADS` | BLAS/thread-pool sizing (8 measured optimal) |
| `RUN_PYTHON_COMPARISON` | enables Julia-vs-Python comparison tests |

A knob is only certified if a test exercises it. There is no `INTFER_*` flag
machinery here — the C++ program's default-OFF-flag census has no counterpart
because the Julia port has almost no runtime knobs.

## Dead ends — measured, do NOT retry as-is

* **Software BF16 as the default path** — Julia lacks AVX-VNNI-BF16 intrinsics;
  software BF16->F32 is ~23x slower than BLAS F32 matmul on Arrow Lake. The
  pipeline exists (`BF16Support.jl`, `ArrowLake.jl`,
  `set_inference_precision!`) but stays off by default.
* **Full GPU forward** — blocked at the driver layer (see READ FIRST). Element-wise
  KA kernels are correct; the matmul precision divergence and the NEO/driver
  versions are the wall. Revisit only when a newer `NEO_jll` or a fixed
  system runtime ships.
* **Gemma4 full-FP32 on GPU** — OOM; the Q4_K-packed streaming structure
  (~289 MB, `Gemma4GPU.jl`) is the working shape, but the "COMPLETE" GPU docs in
  `docs/` disagree about whether weights are placeholders — check the tree, not
  the docs, before trusting either.
* **C++-era dead ends** (SYCL command graphs, native L0 immediate submit,
  host-private DPAS staging, `q.prefetch()`, OpenVINO, DPAS variants,
  MR/SPLITK, fused-multi-GEMM decode extraction) — all measured against the SYCL
  kernel stack, which does not exist here. The full tables live in
  `/var/home/fra/dev/intfer/AGENTS.md`; do not re-run them against Julia without
  a reason.

## Box hardware facts (2x Arc Pro B580)

Source of record: `docs/Intel Arc B580 Specs _ TechPowerUp GPU Database.html`
(operator-saved), cross-checked vs Intel ARK 241598: **20 Xe cores (EUs)**,
160 vector engines / 160 XMX, 12 GB GDDR6 192-bit @19 Gbps => **456 GB/s**
peak bandwidth, 18 MB L2, FP32 13.67 TFLOPS @ 2670 MHz (~2560 FP32 lanes),
256 KB L1 per EU. Xe2 sub-group width is **16**; SLM > 128 KiB per work-group
can **hang** the device. Figures that circulated earlier (512 Xe cores, ~179
GB/s) are the A770's and were never measured on this box — do not reuse.

## History and audit trail

* `HISTORY.md` — Julia phase summaries, fixed-bug list, benchmark results.
* `PERFORMANCE.md` — performance baseline and profiling commands.
* `docs/*.md` — working notes (GPU status reports, MTP plan, BLAS threading);
  `docs/` is gitignored scratch — treat it as volatile.
* `/var/home/fra/dev/intfer/` — the C++ program: its `AGENTS.md` carries every
  C++-era measurement, its `scripts/` the host-driver recovery kit, its
  `reports/` the experiment write-ups.
* **Repo culture:** records are not laundered. A superseded claim gets a
  `[SUPERSEDED ...]` annotation, not a rewrite; a historical record that
  disagrees with today's tree is still a true record of when it was written.

## guidelines

- remember to commit any meaningful change made such that in case of regressions everything can be bisected.
- remember to not litter /tmp with large tmp files, RAM is precious
- try to parallelize work as much as possible, spawn tasks (subagents) in different worktrees that implement a specific optimization experiment, then merge all the experiments in one single batch. Every optimization experiment is gated behind a flag in the cli. The user then runs the benchmarks and reports back the results.
- MANDATORY: every batch of optimization experiments handed to the user for testing MUST be written up as a report file in `reports/` (create the directory if missing; filename `reports/<YYYY-MM-DD>-<batch-slug>.md`). Each report MUST contain, per experiment: (1) the flag + kill switch and default state, (2) what it changes and why it should help, (3) the exact benchmark command to run (and the quiet-box rule load1 < 2), (4) the metrics measured (time, allocations, spread) with the caveat that they are measurements, not promises, (5) the expected direction and what counts as a win/neutral/loss, (6) which correctness gates already passed. NEVER put a number you did not measure in a report — leave a blank `RESULT:` line for the user to fill from a quiet-box run.
