# GPU Optimization Status

## Date

2026-05-29

## Commit

Current HEAD (after vendored oneAPI removal)

## Summary

Vendored oneAPI removed. Now using **upstream oneAPI v2.6.1** + **KA v0.9.41** from Julia registry.
GPU compute is **blocked** on two driver issues. All Float16 model code is structurally complete.

## What Changed

- Removed `vendor/oneAPI/` (was pinned to old jlls)
- Using upstream `oneAPI v2.6.1` from General registry
- `KernelAbstractions` upgraded from `v0.9.39` → `v0.9.41`
- KA v0.9.41 API change: `ndrange` is now **positional**, not keyword
  - Old: `kernel!(a, b; ndrange=n)`
  - New: `kernel!(a, b, (n,))`

## GPU Driver Status

### Works
- Device enumeration: 2 x Intel B580 (0xe20b) found via upstream oneAPI
- Level Zero loader loads from Julia artifacts (not system)
- `oneAPI.devices()` returns 2 GPUs

### Broken — GPU compute

1. **NEO_jll v25.44.36015** (from Julia General registry, pinned by oneAPI.jl v2.6.1):
   - Too old for Battlemage Xe2 architecture (B580)
   - SPIR-V kernel compilation fails: `NEOException`
   - Only supports Alchemist (DG2) and earlier
   - **No newer NEO_jll available in Julia registry**

2. **System intel-compute-runtime v26.18.38308.1** (Fedora 44):
   - Crashes on `zeInit(0)`: `Abort at resource_info.cpp:15`
   - GMM (Graphics Memory Manager) initialization fails with `xe` kernel driver
   - Both direct C calls and Julia `Libdl` calls crash
   - Workaround: `intel-gpu-firmware` was just installed — a **reboot** may fix this

## Completed GPU Code

### Files (all structurally complete, untested)
- `src/GPUCommon.jl` — Type-generic kernels (Float16/Float32), fused attention
- `src/FusedKernels.jl` — GPU-native argmax + sampling, fused MLP gate
- `src/Qwen35GPU.jl` — Full Float16 model with `const E = Float16`
  - Batched QKV + gate+up matmuls
  - GPU-native sampling (eliminates 600KB CPU-GPU copy)
  - Float32 config hyperparameters (prevents underflow)
- `src/Gemma4GPUKernels.jl` — Gemma4 GPU kernels
- `src/Gemma4GPU.jl` — Gemma4 GPU model (requires same NEO fix)

### Known KA incompatibility
N/A anymore — KA v0.9.41 handles ndrange correctly. The previous issue (last array consumed as ndrange) was caused by vendored old KA version.

## To Unblock

1. **Reboot** → test if `intel-gpu-firmware` + driver update fixes system `zeInit` crash
2. If system driver works after reboot → switch oneAPI.jl to use system Level Zero
3. Test kernel launch → verify Float16 inference
4. Benchmark: target 140+ tok/s on 2x B580

## CPU Inference (Fallback)

Working: 14-18 tok/s on Qwen3.5-0.8B Q4_K_M with F32 CPU inference.