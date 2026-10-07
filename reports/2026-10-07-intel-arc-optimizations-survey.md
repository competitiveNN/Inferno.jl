# Intel Arc GPU Optimizations Survey (llama.cpp SYCL, vllm-xpu-kernels, sgl-kernel-xpu)

Date: 2026-10-07
Source: cloned HEAD of all three repos, surveyed source tree

---

## 1. llama.cpp SYCL backend (`ggml/src/ggml-sycl/`)

**Approach: lean on oneDNN/oneMKL for matmul; hand-written subgroup tile kernels for quant path.**

| Optimization | Where | Notes |
|---|---|---|
| XMX/systolic detection | `common.cpp:58` `gpu_has_xmx()` | Probes `sycl::aspect::ext_intel_matrix` |
| XMX hand-written path | commented out (`ggml-sycl.cpp:177`) | `SYCL_USE_XMX` is `#if 0`'d with note "not used for XMX really" |
| oneDNN matmul | `ggml-sycl.cpp:123` → `DnnlGemmWrapper::row_gemm` | Probes `dnnl::matmul` primitive desc; rejects `"ref"` impl. Called for large contiguous matmuls (`ggml-sycl.cpp:3008`, `:3061`) |
| oneMKL BLAS `gemm` | `ggml-sycl.cpp:3115` | `oneapi::mkl::blas::column_major::gemm` fallback |
| Quantized matvec tile kernels | `dmmv.cpp`, `mmq.cpp` | Custom `nd_range` + `reqd_sub_group_size(WARP_SIZE)` kernels; dequant-on-the-fly; `permute_sub_group_by_xor` warp reduction. **No DPAS/XMX** — runs on EU vector ALUs only |
| BF16 | `common.hpp:33` | `sycl::ext::oneapi::bfloat16` |
| Sub-group width = 16 | `common.hpp:226` | Preferred on Intel GPUs |
| SLM | `fattn-tile.cpp`, `fattn-sparse.cpp` | `local_accessor` staging for flash-attention tiles |

**Bottom line:** llama.cpp does **not** hand-write DPAS/XMX. It relies on **oneDNN's internal XMX dispatch** for matmul throughput.

---

## 2. vllm-xpu-kernels (`csrc/xpu/`)

**Approach: CUTLASS-first with architecture-specific (Xe2/Xe3) kernel trees; oneDNN for some quant GEMMs.**

| Optimization | Where |
|---|---|
| CUTLASS DPAS matmul | `XE_DPAS_TT<8, float, ...>` in `grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:104`, `attn/xe_2/paged_decode.hpp:506`, `attn/xe_3/...` |
| CUTLASS BDPAS (Xe3) | `XE_BDPAS_TT` via Xe3 grouped-gemm & attention paths |
| XMX-backed dispatch policies | `MainloopIntelXeXMX16*` (FP8 scaling, block-scaled, fused epilogue with `IntelXeXMX16`) across fp8/mxfp8 paths |
| oneDNN GEMM | `onednn/onednn_matmul.cpp` — w4a4_fp4, w8a16_int4, w8a8_fp8, w4a16_int4 via `dnnl_matmul_*` |
| oneMKL BLAS | via oneDNN wrappers |
| Split-K / TileScheduler | `kernel/xe_tile_scheduler.hpp` — persistent + non-persistent, work-stealing |
| SLM work-stealing | TileScheduler with one `atomicAdd` per tile-chunk |
| Block-2D async prefetch + barrier K-pipeline | `make_block_2d_prefetch` + `prefetch(...)` + `barrier_arrive/wait` in Xe2 kernels |
| Split barriers | `cute::xe_split_barrier` |
| Sub-group SG = 16 | `xe20/w4a16/gemm_xe2.hpp` + all Xe2 kernels |
| Flash decoding (FMHA) | Paged KV, multi-split, per-head-group tiles in `attn/xe_2/`, `attn/xe_3/` |
| FP8/MXFP8/FP4/MXFP4 quantized activation | `quantization/`, `moe/*` |

---

## 3. sgl-kernel-xpu (`src/sycl/`)

**Approach: Similar CUTLASS base as vllm, but no oneDNN; broader kernel surface (MLA sparse, GDN, HiSparse, etc.).**

| Optimization | Where |
|---|---|
| CUTLASS DPAS (Xe2) | `xe20/w4a16/gemm_xe2.hpp` (mainloop: `dpas` referenced explicitly), `xe20/bf16/moe_mainloop.hpp`, `gdn_attn/gemm.hpp` |
| CUTLASS BDPAS (Xe3) | `XE_BDPAS_TT<8, float, ...>` in `xe35/blockwise_moe_mxfp4.cpp:69`, `xe35/fp8_scaled_mm.cpp:114`, `xe35/dsv3_fused_a_gemm.cpp:141` |
| XMX 8×16×16 f16 MAC | `XE_8x16x16_F32F16F16F32_TT` in `xe35/fp8_scaled_mm.cpp:114` |
| Xe2/Xe3 dispatch | Separate `xe20/` vs `xe35/` trees; `torch_extension_sycl.cc:256` selects arch |
| SLM work-stealing tile scheduler | `xe20/w4a16/gemm_xe2.hpp` — SLM-backed atomic counter for tile scheduling |
| Block-2D async prefetch + barrier K-pipeline | `make_block_2d_prefetch` + `prefetch` + `barrier_arrive/wait` in `xe20/w4a16/gemm_xe2.hpp:135-166`, `gdn_attn/gemm.hpp` |
| Split barriers | `cute::xe_split_barrier` in `xe20/w4a16/gemm_xe2.hpp:36` |
| Sub-group shuffle | `sub_group_size = 16` in sampler kernels; `reqd_sub_group_size(32)` in `multimodal_rope.cpp` |
| Flash-attention-v2, MLA (dense + sparse 2-stage), GDN/GatedDeltaNet, HiSparse, MoE, LoRA, MQA logits, HC prefill | `src/sycl/kernels/` tree |
| No oneDNN | Verified: zero `dnnl`/`oneDNN` refs in `src/sycl/` |

---

## 4. Comparison matrix

| Feature | llama.cpp SYCL | vllm-xpu-kernels | sgl-kernel-xpu |
|---|---|---|---|
| DPAS/XMX hand-written | detect-only, unused | CUTLASS `XE_DPAS_TT`/`XE_BDPAS_TT` | CUTLASS `XE_DPAS_TT`/`XE_BDPAS_TT` |
| XMX-backed matmul | oneDNN probe → `DnnlGemmWrapper` | CUTLASS `MainloopIntelXeXMX16*` | CUTLASS `MainloopIntelXeXMX16*` + `XE_8x16x16_*` |
| oneDNN | ✓ (gemm, attention) | ✓ w4a4/w8a8/fp4/fp8 | ✗ (not used) |
| oneMKL BLAS | ✓ column_major::gemm | via oneDNN wrapper | via oneDNN wrapper |
| Sub-group width | 16 | 16 SG | 16 SG (some kernels use 32) |
| SLM work-stealing scheduler | ✗ | ✓ TileScheduler | ✓ SLM counter |
| K-pipeline prefetch+barrier | ✗ | ✓ `make_block_2d_prefetch`+`barrier` | ✓ same |
| Split barriers (`xe_split_barrier`) | ✗ | ✓ | ✓ |
| Async USM / SYCL graphs | ✓ `ext_oneapi_async_memory_alloc` | ✗ | ✗ |
| MoE | ✗ | ✓ | ✓ |
| Flash-attention | sparse fattn | FMHA paged decode/prefill | FMHA v2 + MLA sparse |
| GDN / GatedDeltaNet | ✓ (`gated_delta_net.cpp`) | ✓ | ✓ |
| FP4 / MXFP4 | ✗ | ✗ | ✓ (`mxfp4_blockwise_moe`, etc.) |
| Xe2/Xe3 dispatch | ✗ | ✓ | ✓ |

---

## 5. What cannot be done in pure Julia (oneAPI.jl v2.6.2 + KernelAbstractions v0.9.43)

### Cannot (no Julia binding / KA feature)

1. **DPAS/XMX matrix-core intrinsics** (`XE_DPAS_TT`, `XE_BDPAS_TT`, `XE_8x16x16_*`)
   - Exposed only via `sycl::ext::intel::esimd::matrix` / `sycl_ext_oneapi_matrix`
   - KA emits SPIR-V scalars/vectors; **no `@mma` concept**
   - This is the entire throughput bottleneck for matmul-bound LLM inference

2. **CUTLASS infrastructure** (`TiledMMA`, `block_2d_copy`, `xe_split_barrier`, fused epilogues)
   - C++ template library; no Julia binding exists
   - Replicating in KA would require rewriting tens of thousands of lines of tiled-GEMM machinery

3. **oneDNN primitives** (`dnnl_matmul_w4a4_fp4`, `dnnl_matmul_w8a16_fp8`, etc.)
   - Verified: `isdefined(oneAPI, :oneDNN)` is `false` in oneAPI.jl v2.6.2
   - Only oneMKL is bound (`oneAPI.oneMKL.gemm` ✅)

4. **SLM async-block-prefetch K-pipeline**
   - `make_block_2d_prefetch` + `barrier_arrive/wait` overlap is CUTLASS-specific
   - KA has `@localmem`/`SharedMemory` and `@synchronize`, but no async copy + split-barrier primitive

5. **Portable sub-group shuffle inside `@kernel`**
   - oneAPI.jl exports `sub_group_shuffle_xor` / `sub_group_barrier` at the host level
   - KA has **no `@subgroup`** intrinsic; cannot write shuffle-based reductions in portable KA kernels

6. **FP4/MXFP4 4-bit XMX matmul**
   - Hardware-only path (`XE_BDPAS_TT` on MXFP4); no Julia package has a 4-bit matmul path

7. **GRF pressure tuning** (`sycl::ext::intel::experimental::grf_size_properties`)
   - No Julia hook; C++ repos tune per-thread register allocation explicitly

8. **Async USM + SYCL graphs** (`ext_oneapi_async_memory_alloc`)
   - llama.cpp's async-alloc path; not surfaced by oneAPI.jl

### Already doable (or done in Inferno.jl)

| # | Optimization | Status in Inferno |
|---|---|---|
| 1 | Element-wise kernels (activation, norm, RoPE) | `FusedKernels.jl` `@kernel` ✅ |
| 2 | Chunked reductions + host fold | `FusedKernels.jl` (per-thread ≤ 32, host reduce) ✅ |
| 3 | `oneMKL.gemm` for BF16/FP16/FP32 matmul | `oneAPI.oneMKL.gemm` available; **NOTE:** carries known matmul precision-divergence issue (AGENTS.md: 24-layer forward diverged on matmul precision, Apr 2026) |
| 4 | Shared-mem scratch (SLM) for small tiles | KA `@localmem` / `Scratchpad` ✅ |
| 5 | BF16/FP16 bit reinterpret | Pure Julia (`UInt16 → UInt32<<16 → Float32`) ✅ |
| 6 | CPU quantized matvec | `QuantsCPU.jl`/`QuantMV.jl` ✅ |

---

## 6. Conclusion

**The entire performance gap is the GEMM/matmul tier.** Element-wise, reduction, norm, RoPE, and sampling kernels are either already implemented in `FusedKernels.jl` or trivially portable to KA. The matmul tier — which is where all three repos get their actual token/s throughput — requires either:

- **oneDNN** (llama.cpp's preferred path; not bound in Julia), or
- **CUTLASS + `XE_DPAS_TT`/`XE_BDPAS_TT`** (vllm + sgl; no Julia binding), or
- **`sycl_ext_oneapi_matrix`** (no KA support)

Until a Julia binding to oneDNN/ESIMD-matrix appears, or the NEO/compute-runtime issue on B580 is resolved, the GPU path cannot reach the target throughputs (40 tok/s decode, 4000 tok/s prefill). CPU (BLAS F32) remains the reference backend.

---

## References (file:line within each cloned repo)

| Repo | Key files |
|---|---|
| llama.cpp SYCL | `ggml/src/ggml-sycl/common.cpp:58`, `common.hpp:99,226`, `ggml-sycl.cpp:123,177,3008,3061`, `dmmv.cpp`, `mmq.cpp` |
| vllm-xpu-kernels | `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:104`, `csrc/xpu/attn/xe_2/paged_decode.hpp:506`, `csrc/xpu/attn/xe_3/paged_decode.hpp:508`, `csrc/xpu/onednn/onednn_matmul.cpp` |
| sgl-kernel-xpu | `src/sycl/xe35/fp8_scaled_mm.cpp:69,114-118`, `src/sycl/xe35/blockwise_moe_mxfp4.cpp:69,75-76`, `src/sycl/kernels/moe/xe20/w4a16/gemm_xe2.hpp`, `src/sycl/kernels/moe/xe20/w8a16/moe_mainloop.hpp:155-166`, `src/sycl/kernels/gdn_attn/gemm.hpp` |
