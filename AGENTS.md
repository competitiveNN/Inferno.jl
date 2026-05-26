# Inferno.jl Development Guide

## Current Status: GGUF + Safetensors CPU Inference WORKING + OPTIMIZED

Both GGUF and Safetensors inference for Qwen3.5-0.8B-VL work with coherent generation.

**Verified outputs:**
- "What is 2 + 2 ?" → "\n\n2 + 2 = 4\n\nWhat is 2 + 3 ?\n\n2 + 3 = 5..."
- Matches HuggingFace reference output exactly

**Performance:**
- Per-token allocation: 2.7MB → 10KB (99.6% reduction)
- Throughput: 14-18 tokens/sec

---

## Roadmap

### Phase 1: Fix Safetensors CPU Inference ✓ COMPLETE

**Fixed bugs:**
1. Layer index substring matching (`"layers.1"` matched layers 10, 11, 12...)
2. Attention q_norm/k_norm missing +1 (layernorm1p convention)
3. 3D conv1d tensor handling in get_tensor()
4. Position calculation in generation loop

### Phase 2: Performance Optimizations ✓ COMPLETE

**Achieved:**
- Per-token allocation: 2.7MB → 10KB (99.6% reduction)
- Pre-allocated buffers for all major operations
- Manual @simd loops for slice assignments (0 allocations)
- In-place normalization everywhere
- BLAS operations with pre-allocated output

**Remaining:**
- [x] SIMD vectorization with LoopVectorization.jl
- [x] BLAS threading optimization (currently 10 threads)
- [x] MKL vs OpenBLAS comparison

### Phase 2.6: BF16 Support as Default

**Goal:** Add proper BF16 support for Arrow Lake CPUs (Core Ultra 2 series).

**Status:** RESEARCHED - BF16 pipeline implemented but **disabled by default**.

**Key Finding:** Julia lacks AVX-VNNI-BF16 intrinsics. Without hardware VNNI-BF16 
instructions (VDPBF16PS), BF16 weights require per-element software conversion to F32 
which is 23x slower than BLAS F32 matmul. BF16 saves 50% memory but the software 
conversion cost completely negates the bandwidth savings on CPU.

**What works:**
- `src/ArrowLake.jl`: CPU feature detection (AVX2, AVX-VNNI, Arrow Lake)
- `maybe_bf16()`: Weight conversion at load time (F32 -> BF16, 50% memory)
- `bf16_matmul_vec!()`: Zero-alloc BF16 matmul via UInt16 reinterpret + in-register F32 convert
- `lm_head_project!()` BF16 variant with chunked parallel execution
- `QuantOrFloat32` Union type extended with `Matrix{BFloat16}`
- Auto-detection of Arrow Lake CPUs

**What doesn't work:**
- BF16 inference is 23x slower than F32 (0.04x speedup) because Julia can't emit
  VDPBF16PS instructions - each BF16 element must be converted to F32 in software
- BLAS gemv with F32 is extremely well-optimized; custom @turbo loops are slower

**To enable BF16 for memory-constrained scenarios:**
```julia
model, tok = load_model_cpu("model.gguf"; use_bf16_weights=true)
```

**Future path to BF16 speedup:**
- Wait for Julia to add VNNI-BF16 intrinsics (LLVM supports them)
- Or write a C kernel with `_mm_dpbf16_ps` intrinsic and call via ccall
- Or use GGUF quantization (Q4_K, Q5_K) which IS faster due to better cache utilization

### Phase 2.7: Flash Attention Integration ✓ COMPLETE

**Status:** Flash Attention integrated into `FullAttentionCPU`, **enabled by default**.

**Benchmark Results (head-only, synthetic data):**
| Seq_len | Standard | Flash | Speedup |
|---------|----------|-------|---------|
| 64 | 0.031 ms | 0.003 ms | **8.99x** |
| 512 | 0.316 ms | 0.031 ms | **10.16x** |
| 2048 | 2.730 ms | 0.199 ms | **13.70x** |

**Key Technical Details:**
- **Online Softmax**: Flash Attention uses online softmax computation, avoiding materializing full attention matrix
- **Tiled Computation**: `BLOCK_N=64` for cache-friendly KV cache access
- **Memory Efficiency**: Sequential access pattern through KV cache vs random access
- **Numerically Correct**: Verified against standard attention with < 2e-6 max error

**Implementation:**
- `src/FlashAttention.jl`: Reference implementation with block-based processing
- `FullAttentionCPU`: Added `fa_output_buf` and `use_flash_attention` fields
- Environment toggle: `INFERNO_USE_FLASH_ATTENTION=false` to disable
- Loader: Auto-initializes flash attention buffers for attention layers

**Activation:**
- Flash Attention is **ON by default** for new models
- Disable: `INFERNO_USE_FLASH_ATTENTION=false julia --project=. ...`
- Per-attention layer control via struct field

**Files Changed:**
- `src/ModelCPU.jl`: FullAttentionCPU struct integration
- `src/LoaderCPU.jl`: Flash attention buffer initialization
- `src/Safetensors.jl`: Safetensors loader integration
- `src/FlashAttention.jl`: New flash attention kernel
- `benchmark/flash_attention_bench.jl`: Benchmark suite
- `tests/flash_attention_verification.jl`: Numerical correctness tests

**Notes:**
- Safetensors BF16 dtype=3 conversion already working: `UInt16 -> UInt32<<16 -> Float32`
- GGUF may need separate BF16 handling
- Consider using BFloat16s.jl package for native Julia BF16

### Phase 2.5: Multi-Token Prediction (MTP) ✓ RESEARCHED

**Status:** Infrastructure implemented, but **MTP does NOT work** for Qwen3.5-0.8B.

**Key Finding:** Qwen3.5-0.8B was NOT trained with MTP mask tokens. The `mtp.*` weights
in the safetensors file are for training only, not inference prediction.

**Evidence:**
- Tested multiple mask tokens (BOS, EOS, PAD, Space, Newline) - all produce wrong predictions
- HF implementation has NO references to `mtp.fc`, `mtp.norm`, etc. in forward pass
- The `_mtp_generate` function just appends masks and runs standard forward pass
- Only position 1 predicts correctly, positions 2-4 produce garbage

**Implementation:**
- `generate_mtp_cpu()` - Main MTP generation function (disabled)
- `mtp_predict_step!()` - Predict k tokens (produces incorrect output)
- `mtp_predict_with_head!()` - MTP head approach (produces garbage)
- `stream_to_stdout_cpu()` - `use_mtp=true` flag shows warning and falls back

**To use MTP properly:**
1. Need a model specifically trained with MTP objective (e.g., `Qwen3-4B-Inst-2507-MTP`)
2. Use correct mask token ID (varies by model)
3. Implement verification step for speculative decoding

### Phase 3: Quantization Support

### Phase 2.7: Threading Tuning ✓ COMPLETE

**Findings:** Based on detailed profiling and benchmarking

**BLAS Thread Scaling (20 physical cores):**
| BLAS Threads | ms/token | tok/s | Status |
|--------------|----------|-------|--------|
| 1 | 96.1 | 10.4 | Baseline |
| 4 | 57.7 | 17.3 | Good |
| **8** | **49.0** | **20.4** | **OPTIMAL** |
| 10 | 56.1 | 17.8 | Default (suboptimal) |
| 12 | 56.4 | 17.7 | Slightly worse |
| 20 | 334 | 3.0 | Catastrophic oversubscription |

**Key Insight:** Thread oversubscription destroys performance! At BLAS=20 threads with 20 Julia threads,
all CPU time is spent in thread synchronization overhead.

**Per-Token Breakdown (Qwen3.5-0.8B, Float32):**
| Component | Time | % of Total |
|-----------|------|------------|
| 18 SSM layers | ~21 ms | 35% |
| 6 Attention layers | ~11 ms | 18% |
| **lm_head** | **~15 ms** | **47%** |
| Total | ~47 ms | 100% |

**Optimization Attempts:**

1. **SSM per-head threading with @threads**: ❌ 0.3x slower
   - Sequential: 0.097 ms/head layer
   - Threaded: 0.322 ms/head layer
   - Work per head too small (6μs) to amortize thread overhead

2. **Chunked lm_head with @spawn**: ✅ 1.5x speedup
   - Standard BLAS: 20.5 ms
   - Chunked (8 chunks): 13.4 ms
   - Speedup: 1.54x
   - Implementation: `lm_head_project!` in ModelCPU.jl

3. **@turbo vs BLAS for small matmuls**: ✅ @turbo wins for 128x128
   - @turbo matmul: 0.10 ms
   - BLAS matmul: 0.17 ms
   - Speedup: 1.72x

**Recommendations:**
- Set BLAS threads to 8 for Qwen3.5-0.8B (update: done in code)
- Keep lm_head chunked implementation
- Avoid threading at granularity < 100μs

---

### Phase 2.8: Flash Attention CPU Implementation ✓ COMPLETE

**Status:** Implemented tiled Flash Attention for CPU backend

**Design:** Adapted Flash Attention-2/3 for CPU:
- BLOCK_N = 64 cache blocks
- Online softmax with running statistics (m, l)
- SIMD-friendly inner loops
- No full attention matrix materialization

**Implementation:** `src/FlashAttention.jl` (included in ModelCPU)

**Next Steps:** Benchmark vs standard attention, integrate into FullAttentionCPU

---

### Phase 2.9: Speculative Decoding ✓ COMPLETE

**Status:** Full speculative decoding implementation with draft/target model support

**Algorithm:**
1. Draft model (smaller/faster) generates gamma tokens
2. Target model validates all gamma in single forward pass
3. Accept tokens until first rejection, sample from residual
4. If all accepted, sample one more from target

**Implementation:** `src/SpeculativeDecoding.jl` (included in ModelCPU)

**API:**
```julia
decoder = SpeculativeDecoder(draft_model, target_model, gamma=5)
result, stats = generate_speculative_cpu(decoder, prompt_tokens; max_tokens=100)
# Returns acceptance_rate, speedup_estimate, etc.
```

**Expected Speedup:** 2-3x when draft is 3-5x faster (e.g., 0.5B draft for 7B target)

---

**Goal:** Support Q4, Q5, Q6, Q8 quantizations for GGUF.

**Current state:** F16 (full precision) works. Need to implement:
1. [x] Q4_K_S / Q4_K_M dequantization
2. [x] Q5_K_S / Q5_K_M dequantization
3. [x] Q6_K dequantization
4. [x] Q8_0 dequantization
5. [ ] Test each quantization level for correctness

**Reference:** llama.cpp quantization format in `ggml-common.h`

### Phase 4: Additional Model Architectures

**Goal:** Support more SSM and attention models.

**Targets:**
1. [x] Qwen3 (non-SSM variant)
2. [x] Mamba / Mamba-2
3. [x] RWKV
4. [ ] Jamba (mixture of SSM and attention)

---

## Model Implementation Reference

When implementing model support, ALWAYS check the reference implementation in HuggingFace transformers first:

```
~/.local/lib/python$PYTHON_VERSION/site-packages/transformers/models/qwen3_5/
```

Key files:
- `modeling_qwen3_5.py` - Main model architecture, forward pass logic
- `configuration_qwen3_5.py` - Model configuration and hyperparameters
- `tokenization_qwen3_5.py` - Tokenizer implementation

The transformers implementation is the ground truth for:
1. Layer normalization (RMSNorm vs LayerNorm, +1 bias convention)
2. Attention patterns (RoPE, sliding window, GQA)
3. MLP structure (gate * up, then down)
4. Any architecture-specific quirks (SSM for Mamba-like models)

---

## Debugging with DaemonMode.jl

Use DaemonMode.jl to speed up debugging - avoids REPL startup overhead.

```bash
# Terminal 1: Start daemon
julia --project=. -e 'using DaemonMode; run_daemon()'

# Terminal 2: Run tests
julia --project=. -e 'using DaemonMode; run_job("test/your_test.jl")'
```

---

## Critical: Multi-Token Verification

**NEVER claim success based on single-token tests!**

When verifying inference:
1. Generate at least 64-128 tokens
2. Check output is COHERENT TEXT (not repetition/garbage)
3. Test prompts:
   - "What is 2 + 2 ?" → Should answer "4"
   - "The capital of France" → Should say "Paris"

Single-token matching proves nothing - the model could produce garbage after first token.

---

## Known Bugs & Fixes

### Multi-token Generation Fix (Apr 2026)
**Problem:** Single-token tests passed but multi-token produced garbage.
**Fix:** KV caches must be initialized ONCE before generation loop and passed through each forward call.

WRONG:
```julia
for i in 1:64
    reset_states!(model)
    caches = [init_kv_cache(...) for _ in 1:num_layers]
    forward!(model, [token], 0, caches)  # Always pos=0!
end
```

CORRECT:
```julia
caches = [init_kv_cache(...) for _ in 1:num_layers]
logits = forward!(model, prompt_tokens, 0, caches)
pos = length(prompt_tokens)
for i in 1:64
    token = argmax(logits[:, end])
    logits = forward!(model, [token], pos, caches)  # Increment pos!
    pos += 1
end
```

### Other Fixed Bugs
1. Conv1D weight transpose: GGUF stores (C,K), kernel expects (K,C)
2. RMSNorm +1: Layer norms use layernorm1p convention (+1), SSM norms don't
3. Decay formula: `exp(ssm_a * softplus(a + dt_bias))`
4. L2 norm on Q/K: Qwen3.5 uses L2 norm, not RMSNorm, for attention
5. Safetensors BF16: dtype=3, need `UInt16 -> UInt32<<16 -> Float32`
6. Safetensors alpha/beta transpose: stored as (num_v_heads, hidden_size), was incorrectly transposed to (1024,16) causing DimensionMismatch in mul! - should stay as (16, 1024)
7. **Chat template missing thinking tokens (CRITICAL)**: Qwen3.5 is a hybrid thinking model. The generation prompt MUST include `<think>` and `</think>` tokens. Without them, the model doesn't know whether to think or answer, producing garbled/incoherent output. Default (no-thinking): `<|im_start|>assistant\n<think>\n\n</think>\n\n`. With thinking: `<|im_start|>assistant\n<think>\n`
8. **Missing `<|im_end|>` stop token**: Without stopping on `<|im_end|>`, the model continues generating past end-of-turn, producing extra conversations
