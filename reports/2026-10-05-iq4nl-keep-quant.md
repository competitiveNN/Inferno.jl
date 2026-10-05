# W1 batch — IQ3_S + Qwen3.8 loader (continued): IQ4_NL keep-quantized support + flag gating

**Status:** implemented, gated behind flag, Pkg.test green. Still in progress: W1 session researching
Qwen3.8-specific metadata/architecture details (stalled ~1h, no writes since d63d41b).

## Committed work in this batch
- `d74fcbb` Fix IQ2XS_GRID: regenerate exact 512-entry llama.cpp table (the table had been corrupted
  to 790 entries — silent corruption of IQ2_XS dequant via `grid_idx = (v & 511)+1`); regression tests
  for table invariants + non-zero IQ2_XS dequant (grid entry 2); all other tables verified byte-identical.
- `d63d41b` Add IQ4_NL CPU keep-quantized support: IQ4NL_Matrix + IQ4NL_BLOCK_SIZE,
  `dequantize_iq4_nl`/`dequantize_iq4_nl_into!` (llama.cpp order), ModelCPU
  `mul_quant_mat_vec` + `mlp_mat_vec_mul` for IQ4NL_Matrix, arch-agnostic metadata lookup
  (`qwen3.*` → `arch.*` fallback, enables Qwen3.8 GGUF loading), IQ4_NL tensor extraction (keep-quantized
  and dequant paths) with `keep_quantized` threaded through loaders, IQ4_NL dequant regression tests.
- `01af43e` Remove accidental duplicate `Q8_0_Matrix` definition.
- **Integration (this session):** flag gating for keep-quantized — env var
  `INFERNO_KEEP_QUANTIZED` (default `"0"` = off, the kill switch), matching the repo's
  `INFERNO_NO_QUANT_KERNELS` convention; updated loader docs.

## (1) Flag + kill switch and default state
- Flag: `INFERNO_KEEP_QUANTIZED` (env var). Default `"0"` → **disabled (kill switch off)**.
  Any non-`"0"` value (e.g. `"1"`, `"true"`) enables keeping weights in quantized form in-memory.
- Equivalent explicit opt-in: `load_model(path; keep_quantized=true)` /
  `load_model_cpu(path; keep_quantized=true)`.
- Doc updated: `src/LoaderCPU.jl` (`load_model_cpu` docstring) and `AGENTS.md` env-vars table.

## (2) What it changes and why it should help
- Memory: IQ4_NL weights stay quantized (~4.6 bits/element) instead of full dequant to Float32. For a
  27B-class model this cuts weight memory ~88% (block size 18 bytes/32 elems; on-the-fly dequant inside
  `mul_quant_mat_vec`). Enables loading large IQ4_NL GGUFs (e.g. Qwen3.8-27B) where F32 dequant would
  OOM / thrash.
- Trade-off: the on-the-fly mat-vec is slower than BLAS F32 for most matrix sizes, so default stays off
  (BLAS path selected automatically). Enabled only where memory pressure warrants it.

## (3) Exact benchmark command (quiet box: load1 < 2)
```bash
# warm-up
INFERNO_KEEP_QUANTIZED=0 julia --project=. -e 'using Inferno; m,t=load_model("model.gguf"; backend=:cpu)' >/dev/null

# baseline: F32 (BLAS, faster)
INFERNO_KEEP_QUANTIZED=0 julia --project=. -e '
  using Inferno, BenchmarkTools
  m,t = load_model("model.gguf"; backend=:cpu)
  @time gen = Inferno.generate_text(m,t,"The quick brown fox",max_tokens=128; temperature=0.0f0)'

# keep-quantized: memory-savings path
INFERNO_KEEP_QUANTIZED=1 julia --project=. -e '
  using Inferno, BenchmarkTools
  m,t = load_model("model.gguf"; backend=:cpu)
  @time gen = Inferno.generate_text(m,t,"The quick brown fox",max_tokens=128; temperature=0.0f0)'

# memory peak (both runs): measure RSS before/after load via /proc/self/status VmRSS or
# `using MemoryAssessment; peak = Base.GC.total_bytes()` around load_model().
```

## (4) Metrics measured
| Metric | baseline (F32) | keep-quantized | RESULT: |
|---|---|---|---|
| Load time | | | RESULT: |
| Peak memory (weights) | | | RESULT: |
| Decode throughput (tok/s, median ≥3 runs) | | | RESULT: |
| Allocations/GC pressure | | | RESULT: |

> **Measurement discipline:** exclude warmup; report medians of ≥3 runs with spread; a delta must clear
> run-to-run spread; never write a number you did not measure — fill the RESULT: lines from a quiet-box
> run (load1 < 2 for the whole arm).

## (5) Expected direction / win criteria
- Win: peak memory reduced by ≥50% (ideally ~80-88% on IQ4_NL weights) with ≤5% decode-throughput loss.
- Neutral: memory down, throughput loss in 5-15%.
- Loss: throughput regression >15% with no commensurate memory benefit, or correctness divergence vs the
  F32 path on a known-good model.

## (6) Correctness gates already passed
- `Pkg.test()`: 1076 pass, 0 fail, 3 broken (pre-marked `@test_broken`) on this exact tree.
- IQ4_NL dequant regression tests (tests/unit/core_components.jl): nibbles 0/1/8/15 → -127/-104/1/113,
  scaling by f16 d=1.0/2.0, full byte 0xEB → 38/89.
- IQ2_XS dequant regression (grid entry 2) + table invariants (length/sorted/unique 256/512/1024/256/512).
- IQ3_S: dispatch wired in LoaderCPU.jl; IQ3S_GRID verified 512 sorted/unique (dequant function unchanged
  from prior tree — verified against llama.cpp layout in research phase).
- End-to-end generation pipeline: 11/11 passed.

## Pending
- W1 session still researching Qwen3.8 metadata/architecture (has not written since d63d41b). If it
  produces Qwen3.8-specific changes, merge and re-run the suite; nothing above depends on it.
