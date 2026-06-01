# Inferno.jl History

This file contains older phase summaries, benchmark results, and completed-work context migrated from the main `AGENTS.md` during cleanup.

## Completed CPU Inference Work

### Phase 1: Fix Safetensors CPU Inference

Fixed bugs:
1. Layer index substring matching (`"layers.1"` matched layers 10, 11, 12...)
2. Attention q_norm/k_norm missing +1 (layernorm1p convention)
3. 3D conv1d tensor handling in get_tensor()
4. Position calculation in generation loop

### Phase 2: Performance Optimizations

Achieved:
- Per-token allocation: 2.7MB → 10KB (99.6% reduction)
- Pre-allocated buffers for all major operations
- Manual @simd loops for slice assignments (0 allocations)
- In-place normalization everywhere
- BLAS operations with pre-allocated output

### Phase 2.6: BF16 Support as Default (Research Only)

BF16 pipeline was implemented but remains disabled by default because Julia lacks AVX-VNNI-BF16 intrinsics, and software BF16→F32 conversion is 23x slower than BLAS F32 matmul on Arrow Lake CPUs.

What works:
- `src/ArrowLake.jl`: CPU feature detection (AVX2, AVX-VNNI)
- `maybe_bf16()`: Weight conversion at load time
- `bf16_matmul_vec!()`: Zero-alloc BF16 matmul via UInt16 reinterpret + in-register F32 convert
- `QuantOrFloat32` extended with `Matrix{BFloat16}`

### Phase 2.7: Flash Attention Integration

Flash attention was integrated into `FullAttentionCPU`, enabled by default. Key results:
- BLOCK_N = 64 cache blocks
- Online softmax avoids materializing full attention matrix
- Benchmarks: 8-13.7x speedup vs standard attention at typical sequence lengths

### Phase 2.7: Threading Tuning

Based on profiling:
- BLAS=8 threads is optimal for Qwen3.5-0.8B on 20-core CPUs
- Thread oversubscription past 8 threads degrades performance severely
- lm_head is the bottleneck (~47% of total time); chunked implementation improved throughput ~1.5x

Verified prompt/output examples:
- `"What is 2 + 2 ?"` → `2 + 2 = 4 ...`
- Matches HuggingFace Qwen3.5 reference output exactly

## GPU Inference Summary

GPU work was completed around late May 2026 against an Intel Arc B580 on the oneAPI stack.
- Element-wise KA kernels and a tiled lm_head were verified numerically correct
- A 24-layer hybrid forward pass ultimately diverged due to matmul precision limits in the available GPU path
- CPU inference at 16 tokens/sec was retained as the working backend for this model size

### Multi-token Generation Bug (Apr 2026)

Single-token tests passed but multi-token produced garbage.
Fix: KV caches must be initialized once before generation and incremented correctly across forward calls.

### Other Fixed Bugs

1. Conv1D weight transpose: GGUF stores (C,K), kernel expects (K,C)
2. RMSNorm +1: Layer norms use layernorm1p convention (+1), SSM norms don't
3. Decay formula: `exp(ssm_a * softplus(a + dt_bias))`
4. L2 norm on Q/K: Qwen3.5 uses L2 norm, not RMSNorm, for attention
5. Safetensors BF16: dtype=3, need `UInt16 -> UInt32<<16 -> Float32`
6. Safetensors alpha/beta transpose must not transpose the designated (16, 1024)-shape layout
7. Chat template missing thinking tokens: Qwen3.5 is a hybrid thinking model; generation prompt must include the expected thinking tokens
8. Missing EOS/end-of-turn token: without stopping on it, generation continues past end-of-turn
