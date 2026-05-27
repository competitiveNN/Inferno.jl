# GPU Setup Guide for Inferno.jl

This guide covers setting up GPU inference for Inferno.jl on Intel Arc GPUs under Fedora.

## Prerequisites

- Fedora Linux with `intel-compute-runtime` and `intel-opencl` packages
- Julia 1.10+ with the Inferno.jl project activated

## Installing Intel GPU Drivers

```bash
sudo dnf install intel-compute-runtime intel-opencl intel-level-zero-gpu
sudo modprobe i915
sudo modprobe intel_gpu_top  # optional, for monitoring
```

## Julia oneAPI.jl Setup

From the Julia REPL in the Inferno.jl project directory:

```julia
using Pkg
Pkg.add("oneAPI")
Pkg.build("oneAPI")  # Important: rebuilds oneAPI artifacts
```

## Verify GPU Access

```julia
using oneAPI
oneAPI.devices()
```

You should see your Intel Arc GPU listed (e.g., `Intel(R) Arc(tm) A770 Graphics`).

## Troubleshooting Common Errors

### Error: `UndefVarError: oneapi_gemm_noleading not defined`

This means the oneAPI.jl C library wrapper is not loaded correctly. Common fixes:

1. **Rebuild oneAPI artifacts** (most common fix):
   ```julia
   using Pkg
   Pkg.build("oneAPI")
   ```

2. **Reinstall oneAPI** if build fails:
   ```julia
   using Pkg
   Pkg.rm("oneAPI")
   Pkg.add("oneAPI")
   Pkg.build("oneAPI")
   ```

3. **Check i915 kernel module** is loaded:
   ```bash
   lsmod | grep i915
   ```

4. **Verify device access**:
   ```bash
   ls -la /dev/dri/card*
   ```
   The user running Julia must have read access.

### Error: `INTELBBLA(3,3)` or similar oneAPI BLAS errors

These typically indicate a mismatch between the installed oneAPI runtime and the Julia oneAPI.jl version. Ensure your system packages are up to date and oneAPI.jl is the latest compatible version.

### Performance Issues

Run `intel_gpu_top` in a separate terminal to monitor GPU utilization during benchmarking:
```bash
intel_gpu_top
```

## Running GPU Benchmarks

```bash
# Set your model path
export INFERNO_MODEL=models/Qwen3.5-0.8B-v3_q4_k_m.gguf

# Run benchmark
julia --project=. -O3 benchmark/gpu_benchmark.jl

# Run profile
julia --project=. -O3 benchmark/gpu_profile.jl
```

## Notes

- The `gpu_benchmark.jl` runs a standard generation benchmark (4096 warmup + 128 token generation by default).
- The `gpu_profile.jl` is for performance profiling with a longer prompt and runs full token generation.
- GPU inference targets ~14 tokens/sec on Intel Arc A770 for FP16 models.

## Model Download

Download a GGUF model (e.g., Qwen3.5 0.8B Q4_K_M) into the `models/` directory before running benchmarks.
