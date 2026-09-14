"""
CPU-only inference backend for Inferno.jl

This module provides pure CPU implementations without GPU dependencies.
Supports both GGUF and Safetensors formats with optional quantization support.

# Key Features
- SSM (Gated DeltaNet) and Full Attention layers
- Flash Attention implementation (optional)
- Speculative Decoding
- BF16 support
- Quantized weights (Q4_K, Q5_K, Q6_K, Q8_0)
- Streaming generation

# Exports
## Configuration
- `QwenConfigCPU`: Model configuration struct

## Models
- `QwenModelCPU`: Main CPU model struct
- `RMSNormCPU`, `MLPCPU`: Layer types
- `GatedDeltaNetCPU`: SSM layer
- `FullAttentionCPU`: Attention layer with optional Flash Attention

## Inference
- `forward_cpu!`: Forward pass
- `generate_cpu`: Generate tokens
- `stream_to_stdout_cpu`: Stream to stdout with token stats
- `softmax_sample`: Sample from logits

## State Management
- `init_kv_cache_cpu`: Initialize KV cache
- `reset_states_cpu!`: Reset model state

# Example
```julia
model, tok = load_model_cpu("model.gguf")
output = stream_to_stdout_cpu(model, tok, "Hello, "; max_tokens=50)
```
"""
module ModelCPU

using LinearAlgebra
using Statistics
using LoopVectorization
using ..CommonOps
using ..QuantsCPU
using ..ArrowLake
using ..QuantizedKernels
using BFloat16s
using Printf

export QwenConfigCPU, QwenModelCPU, KVCacheCPU, forward_cpu!, RMSNormCPU, MLPCPU, GatedDeltaNetCPU, FullAttentionCPU, DecoderLayerCPU, RotaryEmbeddingCPU, QuantOrFloat32
export init_kv_cache_cpu, reset_states_cpu!, softmax_sample, generate_cpu, generate_stream_cpu, stream_to_stdout_cpu, generate_with_cache, GenerationState, create_generation_state, generate_batch

# --- Configuration ---
Base.@kwdef struct QwenConfigCPU
 architecture::Symbol = :qwen
 vocab_size::Int = 151936
 hidden_size::Int = 1024
 intermediate_size::Int = 3584
 num_hidden_layers::Int = 24
 num_attention_heads::Int = 8
 num_key_value_heads::Int = 2
 head_dim::Int = 256
 rms_norm_eps::Float32 = 1e-6f0
 rope_theta::Float32 = 10000000.0f0
 max_position_embeddings::Int = 4096
 full_attention_interval::Int = 4
 ssm_inner_size::Int = 2048
 ssm_state_size::Int = 128
 ssm_group_count::Int = 16
 ssm_time_step_rank::Int = 16
 ssm_conv_kernel::Int = 4
 partial_rotary_factor::Float32 = 0.25f0 # Only 25% of head_dim gets rotary
 # MLA (Multi-Head Latent Attention for DeepSeek)
 q_lora_rank::Int = 0
 kv_lora_rank::Int = 0
 qk_rope_head_dim::Int = 0
 qk_nope_head_dim::Int = 0
 v_head_dim::Int = 0
 # Performance options
 use_bf16_weights::Bool = has_arrow_lake_features() # Auto-enable: BF16 lm_head saves 50% memory, matches BLAS speed
end

# Helper functions
sigmoid(x) = 1.0f0 / (1.0f0 + exp(-x))

function sigmoid!(out::AbstractArray, x::AbstractArray)
    @. out = 1.0f0 / (1.0f0 + exp(-x))
    return out
end

# --- RMS Norm ---
struct RMSNormCPU
    weight::Vector{Float32}
    eps::Float32
end

function RMSNormCPU(weight::AbstractArray{Float32}, eps::Float32)
    return RMSNormCPU(vec(weight), eps)
end

@inline function (norm::RMSNormCPU)(x::AbstractArray{Float32})
 ss = mapreduce(abs2, +, x)
 m = ss / length(x)
 scale = 1.0f0 / sqrt(m + norm.eps)
 return x .* scale .* norm.weight
end

@inline function rmsnorm_cpu!(out::AbstractArray{Float32}, x::AbstractArray{Float32}, norm::RMSNormCPU)
 # Compute sum of squares with @turbo for SIMD
 ss = zero(Float32)
 @turbo for i in eachindex(x)
 ss += x[i] * x[i]
 end
 scale = 1.0f0 / sqrt(ss / length(x) + norm.eps)
 # Apply normalization with @turbo
 @turbo for i in eachindex(x)
 out[i] = x[i] * scale * norm.weight[i]
 end
 return out
end

# --- Rotary Position Embedding ---
struct RotaryEmbeddingCPU
 inv_freq::Vector{Float32}
 max_seq_len::Int
 rotary_dim::Int # Number of dimensions that get rotary (partial rotary)
 # Precomputed cos/sin for all positions
 cos_cache::Matrix{Float32}  # (half, max_seq_len)
 sin_cache::Matrix{Float32}  # (half, max_seq_len)
end

function RotaryEmbeddingCPU(head_dim::Int, theta::Float32 = 10000.0f0, max_seq_len::Int = 4096; rotary_dim::Int = head_dim)
 # Only compute inv_freq for the rotary dimensions
 # Formula: inv_freq[i] = 1.0 / (theta ^ (2i / rotary_dim))
 half = div(rotary_dim, 2)
 inv_freq = Float32[1.0 / (theta ^ (2(i-1)/rotary_dim)) for i in 1:half]
 
 # Precompute cos/sin for all positions
 cos_cache = Matrix{Float32}(undef, half, max_seq_len)
 sin_cache = Matrix{Float32}(undef, half, max_seq_len)
 
 for pos in 1:max_seq_len
 for i in 1:half
 freq = inv_freq[i] * (pos - 1)  # 0-indexed positions
 cos_cache[i, pos] = cos(freq)
 sin_cache[i, pos] = sin(freq)
 end
 end
 
 return RotaryEmbeddingCPU(inv_freq, max_seq_len, rotary_dim, cos_cache, sin_cache)
end

function apply_rotary_emb!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU)
 head_dim, num_heads = size(x, 1), size(x, 2)
 half = div(rope.rotary_dim, 2)
 
 # Use precomputed cos/sin (1-indexed position)
 pos_idx = pos + 1  # Convert 0-indexed to 1-indexed
 @inbounds begin
 cos_vals = view(rope.cos_cache, :, pos_idx)
 sin_vals = view(rope.sin_cache, :, pos_idx)
 end
 
 for h in 1:num_heads
 for i in 1:half
 @inbounds begin
 idx1 = i
 idx2 = i + half
 
 x1 = x[idx1, h]
 x2 = x[idx2, h]
 
 c = cos_vals[i]
 s = sin_vals[i]
 
 x[idx1, h] = x1 * c - x2 * s
 x[idx2, h] = x1 * s + x2 * c
 end
 end
 end
 return x
end

# Fused RMSNorm + RoPE for attention (reduces memory passes)
function rmsnorm_rotary!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU, norm::RMSNormCPU)
 head_dim, num_heads = size(x, 1), size(x, 2)
 half = div(rope.rotary_dim, 2)
 weight = norm.weight
 
 # Use precomputed cos/sin (1-indexed position)
 pos_idx = pos + 1  # Convert 0-indexed to 1-indexed
 @inbounds begin
 cos_vals = view(rope.cos_cache, :, pos_idx)
 sin_vals = view(rope.sin_cache, :, pos_idx)
 end
 
 for h in 1:num_heads
 # First: RMSNorm on this head
 sum_sq = 0.0f0
 for i in 1:head_dim
 @inbounds sum_sq += x[i, h] * x[i, h]
 end
 rms = sqrt(sum_sq / head_dim + norm.eps)
 inv_rms = 1.0f0 / rms
 
 # Apply RMSNorm and RoPE in one pass
 for i in 1:head_dim
 @inbounds x[i, h] = x[i, h] * inv_rms * weight[i]
 end
 
 # Now apply RoPE to the first rotary_dim dimensions
 for i in 1:half
 @inbounds begin
 idx1 = i
 idx2 = i + half
 
 x1 = x[idx1, h]
 x2 = x[idx2, h]
 
 c = cos_vals[i]
 s = sin_vals[i]
 
 x[idx1, h] = x1 * c - x2 * s
 x[idx2, h] = x1 * s + x2 * c
 end
 end
 end
 return x
end

# --- KV Cache ---
struct KVCacheCPU
    k::Array{Float32,3}  # (head_dim, n_kv_heads, max_seq)
    v::Array{Float32,3}
end

function init_kv_cache_cpu(config::QwenConfigCPU, max_seq::Int = 4096)
    k = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    v = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    return KVCacheCPU(k, v)
end

function update_kv_cache!(cache::KVCacheCPU, k::Matrix{Float32}, v::Matrix{Float32}, pos::Int)
 # Manual copy to avoid allocation
 for h in 1:size(k, 2)
 for d in 1:size(k, 1)
 cache.k[d, h, pos + 1] = k[d, h]
 end
 end
 for h in 1:size(v, 2)
 for d in 1:size(v, 1)
 cache.v[d, h, pos + 1] = v[d, h]
 end
 end
 return cache
end

# --- MLP ---
# Union type for weight matrices (Float32, Quantized, or BFloat16 with Arrow Lake support)
const QuantOrFloat32 = Union{Matrix{Float32}, Matrix{BFloat16}, Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

struct MLPCPU
 gate_weight::QuantOrFloat32 # (intermediate, hidden) after GGUF reshape+transpose
 up_weight::QuantOrFloat32 # (intermediate, hidden)
 down_weight::QuantOrFloat32 # (hidden, intermediate)
 # Pre-allocated work buffers
 gate_buf::Vector{Float32}  # intermediate_size
 up_buf::Vector{Float32}    # intermediate_size
 hidden_buf::Vector{Float32} # intermediate_size
 output_buf::Vector{Float32} # hidden_size (for down projection output)
end

# MLP call - uses generic forward that handles both Float32 and quantized weights
function (mlp::MLPCPU)(x::Vector{Float32})
 return mlp_forward(mlp, x)
end

# Helper: multiply quantized matrix by vector (row-wise)
# FAST PATH: Use C AVX2 SIMD kernel (quantize input to Q8_K, integer dot products)
# FALLBACK: Pure Julia element-by-element dequantize+multiply

function mul_quant_mat_vec(mat::Q4_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
 if QuantizedKernels.quant_kernels_available()
 QuantizedKernels.q4_k_gemv!(out, mat.data, x, mat.inner_dim, mat.outer_dim)
 return out
 end
 # Julia fallback: use dequantize_q4_k_block! for correct scale unpacking
 fill!(out, 0.0f0)
 block_values = zeros(Float32, 256)
 for row in 1:mat.outer_dim
 sum_val = 0.0f0
 for block in 0:(mat.inner_dim ÷ 256 - 1)
 global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
 block_offset = global_block_idx * QuantsCPU.Q4_K_BLOCK_SIZE + 1
 QuantsCPU.dequantize_q4_k_block!(block_values, mat.data, block_offset)
 for i in 1:256
 col_idx = block * 256 + i
 sum_val += block_values[i] * x[col_idx]
 end
 end
 out[row] = sum_val
 end
 return out
end

function mul_quant_mat_vec(mat::Q5_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
    if QuantizedKernels.quant_kernels_available()
        QuantizedKernels.q5_k_gemv!(out, mat.data, x, mat.inner_dim, mat.outer_dim)
        return out
    end
    # Julia fallback
    fill!(out, 0.0f0)
    block_values = zeros(Float32, 256)
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * QuantsCPU.Q5_K_BLOCK_SIZE + 1
            QuantsCPU.dequantize_q5_k_block!(block_values, mat.data, block_offset)
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

function mul_quant_mat_vec(mat::Q6_K_Matrix, x::Vector{Float32}, out::Vector{Float32})
	if QuantizedKernels.quant_kernels_available()
		QuantizedKernels.q6_k_gemv!(out, mat.data, x, mat.inner_dim, mat.outer_dim)
		return out
	end
	# Julia fallback
	fill!(out, 0.0f0)
    block_values = zeros(Float32, 256)
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        for block in 0:(mat.inner_dim ÷ 256 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 256) + block
            block_offset = global_block_idx * QuantsCPU.Q6_K_BLOCK_SIZE + 1
            QuantsCPU.dequantize_q6_k_block!(block_values, mat.data, block_offset)
            for i in 1:256
                col_idx = block * 256 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

function mul_quant_mat_vec(mat::Q8_0_Matrix, x::Vector{Float32}, out::Vector{Float32})
    if QuantizedKernels.quant_kernels_available()
        QuantizedKernels.q8_0_gemv!(out, mat.data, x, mat.inner_dim, mat.outer_dim)
        return out
    end
    # Julia fallback
    fill!(out, 0.0f0)
    block_values = zeros(Float32, 32)
    for row in 1:mat.outer_dim
        sum_val = 0.0f0
        for block in 0:(mat.inner_dim ÷ 32 - 1)
            global_block_idx = (row - 1) * (mat.inner_dim ÷ 32) + block
            block_offset = global_block_idx * QuantsCPU.Q8_0_BLOCK_SIZE + 1
            QuantsCPU.dequantize_q8_0_block!(block_values, mat.data, block_offset)
            for i in 1:32
                col_idx = block * 32 + i
                sum_val += block_values[i] * x[col_idx]
            end
        end
        out[row] = sum_val
    end
    return out
end

# Generic multiplication for quantized or Float32 weights
function mlp_mat_vec_mul(weight::Matrix{Float32}, x::Vector{Float32})
    return weight * x
end

function mlp_mat_vec_mul(weight::Q4_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q5_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q6_K_Matrix, x::Vector{Float32})
    out = Vector{Float32}(undef, weight.outer_dim)
    return mul_quant_mat_vec(weight, x, out)
end

function mlp_mat_vec_mul(weight::Q8_0_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

# Generic mat-vec multiplication for attention layers (returns new vector)
function mul_quant_or_float32(weight::Matrix{Float32}, x::Vector{Float32})
 return weight * x
end

function mul_quant_or_float32(weight::Q4_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function mul_quant_or_float32(weight::Q5_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function mul_quant_or_float32(weight::Q6_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function mul_quant_or_float32(weight::Q8_0_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

# In-place mat-vec multiplication for all weight types
# Float32: BLAS (always fastest, even for small matrices)
# BFloat16: bulk reinterpret + BLAS
# Quantized: specialized dequant

function mul!(out::Vector{Float32}, weight::QuantOrFloat32, x::Vector{Float32})
    if weight isa Matrix{Float32}
        LinearAlgebra.mul!(out, weight, x)
    elseif weight isa Matrix{BFloat16}
        ArrowLake.bf16_matmul_vec!(out, weight, x)
    else
        mul_quant_mat_vec(weight, x, out)
    end
    return out
end

# Generic MLP forward pass
function mlp_forward(mlp::MLPCPU, x::Vector{Float32})
 # Use pre-allocated buffers
 gate_buf = mlp.gate_buf
 up_buf = mlp.up_buf
 hidden_buf = mlp.hidden_buf
 output_buf = mlp.output_buf
 
 # Gate with SiLU - compute into gate_buf
 mul!(gate_buf, mlp.gate_weight, x)
 # SiLU: x * sigmoid(x) - manual loop to avoid broadcast allocation
 @turbo for i in eachindex(gate_buf)
 g = gate_buf[i]
 gate_buf[i] = g / (1.0f0 + exp(-g))
 end
 
 # Up projection - compute into up_buf
 mul!(up_buf, mlp.up_weight, x)
 
 # Element-wise multiply - compute into hidden_buf
 @turbo for i in eachindex(hidden_buf)
 hidden_buf[i] = gate_buf[i] * up_buf[i]
 end
 
 # Down projection - compute into output_buf and return
 mul!(output_buf, mlp.down_weight, hidden_buf)
 return output_buf
end

# --- GatedDeltaNet (SSM) ---
struct GatedDeltaNetCPU
 index::Int
 
 # Weight matrices (can be Float32 or quantized)
 in_proj::QuantOrFloat32 # (hidden, conv_channels) — projects to Q+K+V
 gate_proj::QuantOrFloat32 # (hidden, d_inner) — SiLU gate
 ssm_out::QuantOrFloat32 # (d_inner, hidden) — output projection
 ssm_conv1d::Matrix{Float32} # (conv_kernel, conv_channels) — transpose for column access
 
 # Alpha/beta projections (GGUF (heads, hidden) → Julia (hidden, heads) after reshape)
 ssm_alpha_weight::Matrix{Float32} # (hidden, num_v_heads)
 ssm_beta_weight::Matrix{Float32} # (hidden, num_v_heads)
 
 # SSM parameters
 ssm_a::Vector{Float32}
 ssm_dt_bias::Vector{Float32}
 ssm_norm::RMSNormCPU
 
 # Dimensions
 num_v_heads::Int
 num_k_heads::Int
 head_k_dim::Int
 head_v_dim::Int
 d_inner::Int
 conv_channels::Int
 conv_kernel::Int
 
 # State buffers
 conv_state::Matrix{Float32}
 h::Array{Float32,3} # Per-group states: (head_v_dim, head_k_dim, num_v_heads)
 
 # Pre-allocated work buffers to avoid GC
 x_conv_buf::Vector{Float32}
 y_all_buf::Vector{Float32}
 alpha_proj_buf::Vector{Float32}
 beta_proj_buf::Vector{Float32}
 # Pre-allocated mat-vec output buffers
 qkv_buf::Vector{Float32}      # in_proj * x output
 z_buf::Vector{Float32}        # gate_proj * x output
 out_buf::Vector{Float32}      # ssm_out * y_all output
 # Per-head loop buffers
 sk_buf::Vector{Float32}       # state * k output (head_v_dim)
 d_buf::Vector{Float32}        # beta * (v - sk) output (head_v_dim)
 y_h_buf::Vector{Float32}      # state * q output (head_v_dim)
 # Per-head normalization buffers
 q_norm_buf::Vector{Float32}
 k_norm_buf::Vector{Float32}
end

function reset_states_cpu!(m::GatedDeltaNetCPU)
 fill!(m.conv_state, 0.0f0)
 fill!(m.h, 0.0f0)
 return nothing
end

function (m::GatedDeltaNetCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
 # 1. Input projections using pre-allocated buffers
 qkv = m.qkv_buf
 z = m.z_buf
 mul!(qkv, m.in_proj, x)
 mul!(z, m.gate_proj, x)
 
 # 2. Update conv state (ring buffer) - use manual loop to avoid slice allocation
 # The slice assignment `m.conv_state[:, 1:3] .= m.conv_state[:, 2:4]` allocates ~74KB
 # Manual loop is allocation-free
if m.conv_kernel > 1
 for j in 1:(m.conv_kernel-1)
  @turbo for i in 1:m.conv_channels
   m.conv_state[i, j] = m.conv_state[i, j+1]
  end
 end
end
# Copy qkv to the last column of conv_state
@turbo for i in 1:m.conv_channels
 m.conv_state[i, m.conv_kernel] = qkv[i]
end
 
 # 3. Compute convolution with fused SiLU activation
 # x_conv[c] = silu(sum_k conv_state[c,k] * ssm_conv1d[c,k])
 # Fused: compute dot product and apply SiLU in one step
 x_conv = m.x_conv_buf
 @turbo for c in 1:m.conv_channels
 # Fused conv + silu: silu(x) = x / (1 + exp(-x))
 v = zero(Float32)
 for k in 1:m.conv_kernel
 v += m.conv_state[c, k] * m.ssm_conv1d[c, k]
 end
 x_conv[c] = v / (1.0f0 + exp(-v))
 end
 
 # 5. Split into Q, K, V
 qk_size = m.head_k_dim * m.num_k_heads
 q_all = reshape(view(x_conv, 1:qk_size), m.head_k_dim, m.num_k_heads)
 k_all = reshape(view(x_conv, qk_size+1:2*qk_size), m.head_k_dim, m.num_k_heads)
 v_all = reshape(view(x_conv, 2*qk_size+1:2*qk_size+m.d_inner), m.head_v_dim, m.num_v_heads)
 
 # 6. Alpha/beta projections (use pre-allocated buffers)
 # ssm_alpha_weight is now (num_v_heads, hidden_size), no transpose needed
 mul!(m.alpha_proj_buf, m.ssm_alpha_weight, x)
 mul!(m.beta_proj_buf, m.ssm_beta_weight, x)
 alpha_proj = m.alpha_proj_buf
 beta_proj = m.beta_proj_buf
 
# 7. Process each head (delta net) - per-group states
# Use pre-allocated buffer
y_all = m.y_all_buf
fill!(y_all, 0.0f0)

eps = 1.0f-6

for h in 1:m.num_v_heads
 g = ((h - 1) % m.num_k_heads) + 1

 qg = view(q_all, :, g)
 kg = view(k_all, :, g)
 vg = view(v_all, :, h)

 # Apply scale factor (llama.cpp uses 1/sqrt(head_k_dim))
 scale = 1.0f0 / sqrt(Float32(m.head_k_dim))

 # L2 normalize q and k with fused scaling (as per llama.cpp ggml_l2_norm)
 # Fused: norm + divide + scale in one pass
 # Use pre-allocated buffers to avoid allocations in loop
 q_normalized = m.q_norm_buf
 k_normalized = m.k_norm_buf
 
# Compute squared sums for L2 norm with @turbo
q_sum_sq = 0.0f0
k_sum_sq = 0.0f0
@turbo for i in 1:m.head_k_dim
 q_sum_sq += qg[i] * qg[i]
 k_sum_sq += kg[i] * kg[i]
end

# Fused normalize + scale in one pass with @turbo
q_norm_val = 1.0f0 / (sqrt(q_sum_sq) + eps) * scale
k_norm_val = 1.0f0 / (sqrt(k_sum_sq) + eps)
@turbo for i in 1:m.head_k_dim
 q_normalized[i] = qg[i] * q_norm_val
 k_normalized[i] = kg[i] * k_norm_val
end

 # Gate values
 alpha_val = clamp(Float64(alpha_proj[h]) + Float64(m.ssm_dt_bias[h]), -20.0, 20.0)
 softplus_alpha = log(1.0 + exp(alpha_val))
 # HF formula: g = ssm_a * softplus(a + dt_bias)
 # ssm_a is already -exp(A_log), so we just multiply
 decay = Float32(m.ssm_a[h] * softplus_alpha)
 # Note: decay should be negative (from -exp(A_log)), and state *= exp(decay) = state * exp(g)
 # But we want to apply exp(g) to the state, so:
 # state .*= exp(decay) where decay = g = ssm_a * softplus_alpha
 # Actually, HF computes g and then the update is: state = state * exp(g) + ...
 # So we need: decay_to_apply = exp(g) = exp(ssm_a * softplus_alpha)
 decay_to_apply = Float32(exp(decay))

 beta_val = clamp(Float64(beta_proj[h]), -20.0, 20.0)
 beta_gate = Float32(1.0 / (1.0 + exp(-beta_val)))

 # State operations (DeltaNet autoregressive)
 # state: [head_v_dim, head_k_dim] = [128, 128]
 state = view(m.h, :, :, h)
 
 # Decay: state *= exp(g) where g = ssm_a * softplus_alpha
 @turbo state .*= decay_to_apply
 
 # sk = state * k (matrix-vector multiply)
 sk = m.sk_buf
 @turbo for i in 1:m.head_v_dim
 s = zero(Float32)
 for j in 1:m.head_k_dim
 s += state[i, j] * k_normalized[j]
 end
 sk[i] = s
 end
 
 # d = beta * (v - sk)
 d = m.d_buf
 @turbo @. d = beta_gate * (vg - sk)
 
 # State update: S = S + d * k' (outer product) - use BLAS.ger! for rank-1 update
 # ger!(alpha, x, y, A) computes A = alpha*x*y' + A
 BLAS.ger!(1.0f0, d, k_normalized, state)
 
 # y = state * q
 y_h = m.y_h_buf
 @turbo for i in 1:m.head_v_dim
 s = zero(Float32)
 for j in 1:m.head_k_dim
 s += state[i, j] * q_normalized[j]
 end
 y_h[i] = s
 end
 
 yg = view(y_all, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
 yg .= y_h
end
# End of loop

    
# 8. Apply SSM norm (per-head normalization)
# y_all has shape (d_inner,) = (head_v_dim * num_v_heads,)
# norm is applied per-head
for h in 1:m.num_v_heads
 y_h = view(y_all, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
 rmsnorm_cpu!(y_h, y_h, m.ssm_norm)
end

# 9. SiLU gate on z
# silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
# Output: norm(y_all) * silu(z) - use @turbo for SIMD
@turbo for i in 1:length(y_all)
 y_all[i] = y_all[i] * z[i] * (1.0f0 / (1.0f0 + exp(-z[i])))
end
# Removed variance scaling as it caused looping/instability
# y_all .*= 1.0f0 / sqrt(Float32(m.head_v_dim))
 
 # 10. Output projection using pre-allocated buffer
 out = m.out_buf
 mul!(out, m.ssm_out, y_all)
 return out
end

# --- SSM Matrix-Vector Multiplication Helpers ---
# These functions handle both Float32 and quantized weight matrices

function ssm_mat_vec_mul(weight::Matrix{Float32}, x::Vector{Float32})
 return weight * x
end

function ssm_mat_vec_mul(weight::Q4_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q5_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q6_K_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

function ssm_mat_vec_mul(weight::Q8_0_Matrix, x::Vector{Float32})
 out = Vector{Float32}(undef, weight.outer_dim)
 return mul_quant_mat_vec(weight, x, out)
end

# --- Full Attention ---
struct FullAttentionCPU
 index::Int
 wq::QuantOrFloat32 # (n_heads * head_dim * 2, hidden) — query + gate projection
 wk::QuantOrFloat32 # (n_kv * head_dim, hidden)
 wv::QuantOrFloat32 # (n_kv * head_dim, hidden)
 wo::QuantOrFloat32 # (hidden, n_heads * head_dim) — output projection
 q_norm::RMSNormCPU
 k_norm::RMSNormCPU
 n_heads::Int
 n_kv::Int
 head_dim::Int
 scale::Float32
 # Pre-allocated work buffers
 qkv_buf::Vector{Float32} # wq output (n_heads * head_dim * 2)
 k_buf::Vector{Float32} # wk output (n_kv * head_dim)
 v_buf::Vector{Float32} # wv output (n_kv * head_dim)
 query_states_buf::Vector{Float32} # query after split (n_heads * head_dim)
 gate_buf::Vector{Float32} # gate after split (n_heads * head_dim)
 output_buf::Vector{Float32} # final output (n_heads * head_dim)
 scores_buf::Vector{Float32} # attention scores (max_seq_len)
 wo_output_buf::Vector{Float32} # wo output (hidden_size)
 # Flash attention buffers (reused per head)
 fa_output_buf::Vector{Float32} # (head_dim) for flash attention accumulation
 use_flash_attention::Bool
end

function (attn::FullAttentionCPU)(x::Vector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
 # Q, K, V projections using pre-allocated buffers
 mul!(attn.qkv_buf, attn.wq, x)
 mul!(attn.k_buf, attn.wk, x)
 mul!(attn.v_buf, attn.wv, x)

 # Split qkv into query and gate - manual loop to avoid allocation
 query_states = attn.query_states_buf
 gate = attn.gate_buf
 
 for h in 1:attn.n_heads
 base = (h - 1) * (2 * attn.head_dim)
 for i in 1:attn.head_dim
 query_states[(h-1)*attn.head_dim+i] = attn.qkv_buf[base+i]
 gate[(h-1)*attn.head_dim+i] = attn.qkv_buf[base+attn.head_dim+i]
 end
 end

 # Reshape to (head_dim, num_heads)
 query_states = reshape(query_states, attn.head_dim, attn.n_heads)
 k = reshape(attn.k_buf, attn.head_dim, attn.n_kv)
 v = reshape(attn.v_buf, attn.head_dim, attn.n_kv)

 # Fused RMSNorm + RoPE for Q and K (reduces memory passes)
 rmsnorm_rotary!(query_states, pos, rope, attn.q_norm)
 rmsnorm_rotary!(k, pos, rope, attn.k_norm)

 # Update KV cache
 update_kv_cache!(cache, k, v, pos)

 # Compute attention: choose between standard and flash attention
 output = attn.output_buf
 fill!(output, 0.0f0)

 gqa_ratio = div(attn.n_heads, attn.n_kv)
 seq_len = pos + 1

 for h in 1:attn.n_heads
 kv_h = div(h - 1, gqa_ratio) + 1
 q_h = view(query_states, :, h)

 if attn.use_flash_attention
 # Flash Attention: memory-efficient tiled computation
 fa_out = view(attn.fa_output_buf, 1:attn.head_dim)
 fill!(fa_out, 0.0f0)
 
 # Call the flash_attention_cpu! function from FlashAttention.jl
 flash_attention_cpu!(fa_out, q_h, cache.k, cache.v, kv_h, seq_len, attn.scale, attn.head_dim)
 
 # Copy result to output buffer
 @turbo for i in 1:attn.head_dim
 output[(h-1)*attn.head_dim+i] = fa_out[i]
 end
 else
 # Standard attention with manual @turbo
 K_h = view(cache.k, :, kv_h, 1:seq_len)
 V_h = view(cache.v, :, kv_h, 1:seq_len)
 
 # Compute scores: K' * q = (seq_len, head_dim) * (head_dim,) = (seq_len,)
 scores = view(attn.scores_buf, 1:seq_len)
 @turbo for i in 1:seq_len
 s = zero(Float32)
 for j in 1:attn.head_dim
 s += K_h[j, i] * q_h[j]
 end
 scores[i] = s * attn.scale
 end

 # Softmax (in-place on scores buffer)
 max_score = maximum(scores)
 @turbo for i in 1:seq_len
 scores[i] = exp(scores[i] - max_score)
 end
 sum_scores = sum(scores)
 @turbo for i in 1:seq_len
 scores[i] /= sum_scores
 end

 # Weighted sum: V * scores = (head_dim, seq_len) * (seq_len,) = (head_dim,)
 @turbo for i in 1:attn.head_dim
 s = zero(Float32)
 for j in 1:seq_len
 s += V_h[i, j] * scores[j]
 end
 output[(h-1)*attn.head_dim+i] = s
 end
 end
 end

 # Apply sigmoid gate to output (gate comes from Q projection)
 @turbo for i in 1:length(gate)
 gate[i] = 1.0f0 / (1.0f0 + exp(-gate[i]))
 end
 @turbo for i in 1:length(output)
 output[i] *= gate[i]
 end

 # Output projection using pre-allocated buffer
 mul!(attn.wo_output_buf, attn.wo, output)
 return attn.wo_output_buf
end

# --- Decoder Layer ---
struct DecoderLayerCPU
 in_norm::RMSNormCPU
 op::Union{GatedDeltaNetCPU,FullAttentionCPU}
 post_norm::RMSNormCPU
 mlp::MLPCPU
 is_ssm::Bool
 # Pre-allocated norm buffers to avoid allocation
 norm_buf1::Vector{Float32}
 norm_buf2::Vector{Float32}
end

function (layer::DecoderLayerCPU)(x::AbstractVector{Float32}, pos::Int, rope::RotaryEmbeddingCPU, cache::KVCacheCPU)
 # Input normalization - use pre-allocated buffer
 x_norm = layer.norm_buf1
 rmsnorm_cpu!(x_norm, x, layer.in_norm)
 
 # Attention/SSM
 residual = layer.op(x_norm, pos, rope, cache)
 
 # Residual connection (mutate x directly)
 x .+= residual
 
 # Post-attention normalization - use pre-allocated buffer
 x_norm2 = layer.norm_buf2
 rmsnorm_cpu!(x_norm2, x, layer.post_norm)
 
 # MLP
 residual = layer.mlp(x_norm2)
 
 # Final residual
 x .+= residual
 return x
end

# --- MTP (Multi-Token Prediction) Head ---

"""
 MTPHeadCPU

Multi-Token Prediction head for speculative decoding. This is an additional
layer that predicts multiple future tokens in parallel.

The MTP head consists of:
- pre_fc_norm_embedding: RMSNorm for the input embedding
- pre_fc_norm_hidden: RMSNorm for the hidden state
- fc: Combined projection (hidden -> hidden)
- layers: A single attention layer (optional)
- norm: Final normalization

The MTP forward pass:
1. Take the last hidden state h and the embedding of the predicted token e
2. norm(e) + norm(h) -> concatenated
3. fc(concatenated) -> hidden
4. norm(hidden) -> hidden
5. lm_head(hidden) -> logits for next token prediction
"""
struct MTPHeadCPU
 pre_fc_norm_embedding::RMSNormCPU
 pre_fc_norm_hidden::RMSNormCPU
 fc::Matrix{Float32} # (hidden_size, 2*hidden_size)
 layers::Vector{DecoderLayerCPU} # Optional MTP layers (typically 1)
 norm::RMSNormCPU
 # Pre-allocated buffers
 embed_buf::Vector{Float32}
 hidden_buf::Vector{Float32}
 combined_buf::Vector{Float32}
 fc_out_buf::Vector{Float32}
 logits_buf::Vector{Float32}
end

# --- Full Model ---
struct QwenModelCPU
 config::QwenConfigCPU
 embed::Matrix{Float32} # (hidden, vocab_size)
 lm_head::Union{Matrix{Float32}, Matrix{BFloat16}} # (vocab_size, hidden) or tied to embed
 layers::Vector{DecoderLayerCPU}
 final_norm::RMSNormCPU
 rope::RotaryEmbeddingCPU
 # Pre-allocated buffers to avoid allocations per token
 embed_buf::Vector{Float32} # embedding lookup buffer (avoids slice copy)
 final_norm_buf::Vector{Float32}
 lm_head_buf::Vector{Float32}
 # Optional MTP head (for models trained with Multi-Token Prediction)
 mtp::Union{MTPHeadCPU, Nothing}
end

function forward_cpu!(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU}; full_logits::Bool=false)
 seq_len = length(tokens)

 if full_logits
 all_logits = zeros(Float32, model.config.vocab_size, seq_len)
 for t in 1:seq_len
 tok = tokens[t] # Already 1-indexed from encode()
 curr_pos = pos + t - 1
# Copy embedding to pre-allocated buffer
@turbo for i in 1:length(model.embed_buf)
 model.embed_buf[i] = model.embed[i, tok]
end
 x = model.embed_buf
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 # Use pre-allocated buffers for final_norm and lm_head
 # Project to vocab (use optimized blocked multiply)
 lm_head_project!(model.lm_head_buf, model.lm_head, model.final_norm_buf)
 all_logits[:, t] = model.lm_head_buf
 end
 return all_logits
 else
 # Only compute LM head for the last position
 last_logits = model.lm_head_buf
 for t in 1:seq_len
 tok = tokens[t] # Already 1-indexed from encode()
 curr_pos = pos + t - 1
# Copy embedding to pre-allocated buffer (avoids slice allocation)
@turbo for i in 1:length(model.embed_buf)
 model.embed_buf[i] = model.embed[i, tok]
end
 x = model.embed_buf
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 if t == seq_len
 # Use pre-allocated buffers for final_norm and lm_head
 rmsnorm_cpu!(model.final_norm_buf, x, model.final_norm)
 lm_head_project!(last_logits, model.lm_head, model.final_norm_buf)
 end
 end
 return reshape(last_logits, model.config.vocab_size, 1)
 end
end

"""
    GenerationState

Persistent generation state with pre-allocated KV caches.
Avoids ~800MB allocation per generation call by reusing caches.

# Example
```julia
state = create_generation_state(model; max_context=8192)
output = generate_with_cache(model, tokenizer, "Hello", state; max_tokens=50)
# Continue generation with same state
more = generate_with_cache(model, tokenizer, "World", state; max_tokens=20)
```
"""
mutable struct GenerationState
    caches::Vector{KVCacheCPU}
    max_seq::Int
    token_counts::Dict{Int,Int}
    last_token::Int
    curr_pos::Int
    tokens_generated::Int
end

function create_generation_state(model::QwenModelCPU; max_context::Int=8192)
    max_cache_seq = min(model.config.max_position_embeddings, max_context)
    caches = [init_kv_cache_cpu(model.config, max_cache_seq) for _ in 1:model.config.num_hidden_layers]
    reset_states_cpu!(model)
    return GenerationState(caches, max_cache_seq, Dict{Int,Int}(), 0, 0, 0)
end

function reset_state!(state::GenerationState, model::QwenModelCPU)
    reset_states_cpu!(model)
    state.token_counts = Dict{Int,Int}()
    state.last_token = 0
    state.curr_pos = 0
    state.tokens_generated = 0
    return state
end

function reset_states_cpu!(model::QwenModelCPU)
    for layer in model.layers
        if layer.is_ssm
            reset_states_cpu!(layer.op)
        end
    end
end

# --- Sampling Functions ---

"""
 lm_head_project!(output, weight, hidden; nchunks=4)

Optimized large matrix-vector multiply for lm_head projection.

Uses row-wise parallel chunking with spawn+wait to minimize allocations.
For a (vocab_size, hidden_size) matrix with vocab_size >> hidden_size
(e.g., 248K x 1024), splitting into chunks allows each thread to work
on a cache-friendly subset of rows.

Benchmark: 13ms vs 27ms for direct BLAS (2.1x speedup with 4 threads)
Allocations: 1872 bytes vs 8128 bytes for Threads.@threads
"""
function lm_head_project!(output::Vector{Float32}, weight::Matrix{Float32}, hidden::Vector{Float32}; nchunks::Int=4)
 vocab_size = size(weight, 1)
 chunk_size = cld(vocab_size, nchunks)
 
 # Pre-allocate temporary buffers for each chunk
 chunk_outputs = [Vector{Float32}(undef, chunk_size) for _ in 1:nchunks]
 
 # Use spawn+wait instead of @threads to reduce allocations
 tasks = Vector{Task}(undef, nchunks)
 for chunk in 1:nchunks
 i_start = (chunk - 1) * chunk_size + 1
 i_end = min(chunk * chunk_size, vocab_size)
 actual_size = i_end - i_start + 1
 
 tasks[chunk] = Threads.@spawn begin
 buf = view(chunk_outputs[chunk], 1:actual_size)
 BLAS.gemv!('N', 1.0f0, view(weight, i_start:i_end, :), hidden, 0.0f0, buf)
 end
 end
 
 # Wait for all tasks to complete and copy results
 for task in tasks
 wait(task)
 end
 
 # Copy results back to output
 for chunk in 1:nchunks
 i_start = (chunk - 1) * chunk_size + 1
 i_end = min(chunk * chunk_size, vocab_size)
 actual_size = i_end - i_start + 1
 output[i_start:i_end] .= chunk_outputs[chunk][1:actual_size]
 end
end

"""
 lm_head_project! for BFloat16 weights.
 Weight is stored transposed: (hidden_size, vocab_size) so each logical "row"
 (vocab entry) maps to a Julia column, which IS contiguous in memory.
 Uses C AVX2 BF16 kernel for each contiguous column dot product.
"""
function lm_head_project!(output::Vector{Float32}, weight::Matrix{BFloat16}, hidden::Vector{Float32}; nchunks::Int=4)
 # weight is (hidden_size, vocab_size), stored transposed
 # output[j] = dot(weight[:,j], hidden) = sum_i weight[i,j] * hidden[i]
 vocab_size = size(weight, 2)
 
 W_u16 = reinterpret(UInt16, weight)
 
 # Convert F32 activations to BF16 once
 x_bf16 = Vector{UInt16}(undef, length(hidden))
 ArrowLake.fp32_to_bf16_c!(x_bf16, hidden)
 
 # Each column weight[:,j] is contiguous (stride 1) in column-major
 # Use bf16_gemv_c! with the transposed view:
 # Call with W_u16' conceptually, but physically we just pass the whole matrix
 # and let the C kernel iterate over columns.
 # Actually simpler: call bf16_gemv_c! on the full weight
 # bf16_gemv_c! computes out[i] = dot(W_u16[i,:], x_bf16) for row i
 # But we want out[j] = dot(W_u16[:,j], x_bf16) for column j
 # So we use the GEMV kernel on the transposed layout.
 # 
 # The simplest correct approach: use the C kernel's bf16_dot_avx2 directly
 # for each column. But we need a Julia wrapper.
 # 
 # Alternative: just use the already-working bf16_gemv_c! on contiguous rows
 # by reinterpreting the weight matrix layout.
 # 
 # Simplest: use bf16_gemv_c! with (vocab_size, hidden_size) row-major view
 # Since weight is (hidden_size, vocab_size) in Julia column-major,
 # reinterpreting as (vocab_size, hidden_size) row-major would require
 # transposing the data.
 #
 # ACTUALLY: the transposed weight already makes columns contiguous.
 # A column view(W_u16, :, j) IS contiguous and IS what we need for dot product.
 # We just need a Julia-callable bf16_dot_avx2 wrapper.
 
 chunk_size = cld(vocab_size, nchunks)
 chunk_outputs = [Vector{Float32}(undef, chunk_size) for _ in 1:nchunks]
 tasks = Vector{Task}(undef, nchunks)
 for chunk in 1:nchunks
 j_start = (chunk - 1) * chunk_size + 1
 j_end = min(chunk * chunk_size, vocab_size)
 actual_size = j_end - j_start + 1
 
 tasks[chunk] = Threads.@spawn begin
 # Use the GEMV kernel: treat each column as a "row" for the dot product
 # Pass a view of contiguous columns to the C kernel
 # The C kernel bf16_gemv_avx2(out, A, x, m, n, stride) computes:
 # out[i] = dot(A[i*stride .. i*stride+n-1], x)
 # We want: out[k] = dot(W_u16[:, j_start+k-1], x_bf16)
 # Column j of W_u16 starts at W_u16[1,j] = parent[offset + (j-1)*hidden_size]
 # So we can pass columns as if they were rows with stride=hidden_size
 ArrowLake.bf16_gemv_c_cols!(
 view(chunk_outputs[chunk], 1:actual_size),
 W_u16, x_bf16, j_start, j_end, size(weight, 1))
 end
 end
 
 for task in tasks
 wait(task)
 end
 
 for chunk in 1:nchunks
 j_start = (chunk - 1) * chunk_size + 1
 j_end = min(chunk * chunk_size, vocab_size)
 actual_size = j_end - j_start + 1
 output[j_start:j_end] .= chunk_outputs[chunk][1:actual_size]
 end
end

# Use CommonOps.softmax_sample instead of local copy
const softmax_sample = CommonOps.softmax_sample
function apply_presence_penalty!(logits::Vector{Float32}, token_counts::Dict{Int,Int}, penalty::Float32)
 if penalty == 0.0f0
  return
 end
 for (tokid, _cnt) in token_counts
  if 1 <= tokid <= length(logits)
   logits[tokid] -= penalty
  end
 end
end

# Use CommonOps.apply_repetition_penalty! instead of local copy
const apply_repetition_penalty! = CommonOps.apply_repetition_penalty!

# --- Generation Functions ---

"""
    generate_cpu(model, tokens, pos, caches; kwargs...)

Generate a single token from the model.

Returns: (next_token, updated_logits)
"""
function generate_cpu(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU};
    temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0,
    repetition_penalty::Float32=1.0f0, token_counts::Dict{Int,Int}=Dict{Int,Int}(), presence_penalty::Float32=0.0f0, min_p::Float32=0.0f0)
    
    # Forward pass
    logits = forward_cpu!(model, tokens, pos, caches)
    
    # Get logits for last token
    logits_vec = vec(logits[:, end])
    
    # Apply presence then repetition penalties
    apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
    apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
    
    # Sample
    next_token = softmax_sample(logits_vec; temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
    
    return next_token, logits_vec
end

"""
 generate_stream_cpu(model, prompt_tokens; kwargs...)

Create a channel that yields decoded token strings as they are generated.

Usage:
```julia
stream = generate_stream_cpu(model, prompt_tokens; max_tokens=100)
for token_str in stream
 print(token_str)
end
```

MTP Support:
Set `use_mtp=true` to enable Multi-Token Prediction for faster generation.
When enabled, the model predicts `k_toks` tokens per forward pass.

```julia
stream = generate_stream_cpu(model, prompt_tokens; use_mtp=true, k_toks=4)
```
"""
function generate_stream_cpu(model::QwenModelCPU, prompt_tokens::Vector{Int}, decode_fn::Function;
  max_tokens::Int=512,
  temperature::Float32=1.0f0,
  top_p::Float32=0.95f0,
  top_k::Int=0,
  repetition_penalty::Float32=1.0f0,
  presence_penalty::Float32=0.0f0,
  min_p::Float32=0.0f0,
  stop_tokens::Set{Int}=Set{Int}(),
  max_context::Int=8192,
  show_tps::Bool=false,
  io::IO=stdout,
  # MTP options
  use_mtp::Bool=false,
  k_toks::Int=4,
  mask_id::Int=151643,
  # Interrupt support
  interrupt_check::Function=() -> false)
 
 return Channel{String}(32) do chan
 try
 # Initialize caches with reasonable max_seq to avoid OOM
 max_cache_seq = min(model.config.max_position_embeddings, max_context)
 caches = [init_kv_cache_cpu(model.config, max_cache_seq) for _ in 1:model.config.num_hidden_layers]
 reset_states_cpu!(model)
 
 # Track token counts for repetition/presence penalty
 token_counts = Dict{Int,Int}()
 for t in prompt_tokens
 token_counts[t] = get(token_counts, t, 0) + 1
 end
 
 # Process prompt tokens
 curr_pos = 0
 if !isempty(prompt_tokens)
 # Process all prompt tokens together to properly update states
 # logits has shape (vocab_size, length(prompt_tokens))
 logits = forward_cpu!(model, prompt_tokens, 0, caches)
 curr_pos = length(prompt_tokens)

 # The logits for the first generated token are the logits of the last prompt token
 first_logits_vec = vec(logits[:, end])

 # Apply penalties to the first token's logits
 apply_presence_penalty!(first_logits_vec, token_counts, presence_penalty)
 apply_repetition_penalty!(first_logits_vec, token_counts, repetition_penalty)

 # Sample the first token
 next_token = softmax_sample(first_logits_vec; temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)

 curr_pos += 1
 token_counts[next_token] = get(token_counts, next_token, 0) + 1

 # Decode and yield
 token_str = decode_fn([next_token])
 put!(chan, token_str)

 last_token = next_token
 
 # Check if MTP is requested but not supported
 mtp_enabled = use_mtp
 if mtp_enabled
 if model.mtp === nothing
 @warn "MTP requested but model does not have MTP weights. Falling back to sequential generation."
 mtp_enabled = false
 else
 @warn "MTP is experimental. Qwen3.5-0.8B was not trained with MTP mask tokens. Predictions may be incorrect."
 # For now, disable MTP since it produces garbage output
 # TODO: Implement proper MTP when we have a model trained for it
 mtp_enabled = false
 end
 end
 
 # Generate remaining tokens
 tokens_generated = 1
 while tokens_generated < max_tokens
 # Standard single-token generation (MTP disabled for now)
 next_token, _ = generate_cpu(model, [last_token], curr_pos, caches;
 temperature=temperature, top_p=top_p, top_k=top_k, 
 repetition_penalty=repetition_penalty, token_counts=token_counts, 
 presence_penalty=presence_penalty, min_p=min_p)
 
 # Check stop token BEFORE updating state and yielding
 if next_token in stop_tokens
 break
 end
 
 curr_pos += 1
 token_counts[next_token] = get(token_counts, next_token, 0) + 1
 
token_str = decode_fn([next_token])
  put!(chan, token_str)
  
  last_token = next_token
  tokens_generated += 1

  if interrupt_check()
  break
  end
  end
  end
 
catch e
  if !(e isa InvalidStateException) && !(e isa InterruptException)
  @error "ERROR during CPU generation stream" exception=(e, catch_backtrace())
  end
 finally
 try
 close(chan)
 catch
 end
 end
 end
 end

 """
    generate_with_cache(model, tokenizer, prompt, state::GenerationState; kwargs...)

 Generate text using pre-allocated KV caches from `GenerationState`.
 This avoids the ~800MB allocation per call that `generate_text` incurs.

 # Arguments
 - `model`: The QwenModelCPU
 - `tok`: Tokenizer (SimpleTokenizer or BPETokenizer)
 - `prompt`: Text prompt
 - `state`: Pre-allocated GenerationState with KV caches

 # Keyword Arguments
 - `max_tokens`: Maximum tokens to generate (default: 256)
 - `temperature`: Sampling temperature (default: 0.7)
 - `top_p`: Nucleus sampling threshold (default: 0.9)
 - `top_k`: Top-k filtering, 0 to disable (default: 40)
 - `repetition_penalty`: Penalty for repeated tokens (default: 1.1)
 - `stop_tokens`: Set of stop token IDs (default: includes EOS)

 # Example
 ```julia
 state = create_generation_state(model; max_context=8192)
 output1 = generate_with_cache(model, tok, "Hello", state; max_tokens=20)
 # Continue generation with same state (reuses KV caches)
 output2 = generate_with_cache(model, tok, "World", state; max_tokens=20)
 ```
 """
 function generate_with_cache(model::QwenModelCPU, tok, prompt::String, state::GenerationState;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    stop_tokens::Set{Int}=Set{Int}())
    
    push!(stop_tokens, tok.eos_id)
    
    prompt_tokens = getfield(parentmodule(typeof(tok)), :encode)(tok, prompt)
    if isempty(prompt_tokens)
        return ""
    end
    
    decode_fn = (ids) -> getfield(parentmodule(typeof(tok)), :decode)(tok, ids)
    
    return _generate_from_state(model, prompt_tokens, decode_fn, state;
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, stop_tokens=stop_tokens)
    end

 function _generate_from_state(model::QwenModelCPU, prompt_tokens::Vector{Int}, decode_fn::Function,
    state::GenerationState; max_tokens::Int=256, temperature::Float32=1.0f0,
    top_p::Float32=0.95f0, top_k::Int=0, repetition_penalty::Float32=1.0f0,
    presence_penalty::Float32=0.0f0, min_p::Float32=0.0f0,
    stop_tokens::Set{Int}=Set{Int}())
    
    caches = state.caches
    token_counts = state.token_counts
    
    # Reset state tracking
    reset_state!(state, model)
    
    # Process prompt tokens
    curr_pos = 0
    all_tokens = Int[]
    
    if !isempty(prompt_tokens)
        logits = forward_cpu!(model, prompt_tokens, 0, caches)
        curr_pos = length(prompt_tokens)
        
        first_logits_vec = vec(logits[:, end])
        apply_presence_penalty!(first_logits_vec, token_counts, presence_penalty)
        apply_repetition_penalty!(first_logits_vec, token_counts, repetition_penalty)
        
        next_token = softmax_sample(first_logits_vec; temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        
        curr_pos += 1
        token_counts[next_token] = get(token_counts, next_token, 0) + 1
        push!(all_tokens, next_token)
        
        state.last_token = next_token
        state.curr_pos = curr_pos
        state.tokens_generated = 1
        
        if next_token in stop_tokens
            return decode_fn(all_tokens)
        end
    end
    
    # Generate remaining tokens
    while state.tokens_generated < max_tokens
        next_token, _ = generate_cpu(model, [state.last_token], state.curr_pos, caches;
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, token_counts=token_counts,
            presence_penalty=presence_penalty, min_p=min_p)
        
        if next_token in stop_tokens
            break
        end
        
        state.curr_pos += 1
        token_counts[next_token] = get(token_counts, next_token, 0) + 1
        state.last_token = next_token
        state.tokens_generated += 1
        push!(all_tokens, next_token)
    end
    
    return decode_fn(all_tokens)
 end

 """
    generate_batch(model, tokenizer, prompts, state::GenerationState; kwargs...)

 Generate text for multiple prompts sequentially using shared KV caches.
 Pre-allocated cache buffers are reused across all prompts, avoiding repeated
 allocation of ~400MB+ per call. Each prompt is processed one at a time.

 # Arguments
 - `model`: The QwenModelCPU
 - `tok`: Tokenizer (SimpleTokenizer or BPETokenizer)
 - `prompts`: Vector of prompt strings
 - `state`: Pre-allocated GenerationState with KV caches

 # Keyword Arguments
 - `max_tokens`: Maximum tokens to generate per prompt (default: 256)
 - `temperature`: Sampling temperature (default: 0.7)
 - `top_p`: Nucleus sampling threshold (default: 0.9)
 - `top_k`: Top-k filtering, 0 to disable (default: 40)
 - `repetition_penalty`: Penalty for repeated tokens (default: 1.1)

 # Example
 ```julia
 state = create_generation_state(model; max_context=8192)
 prompts = ["Hello", "What is AI?", "Tell me a story"]
 outputs = generate_batch(model, tok, prompts, state; max_tokens=20)
 ```
 """
 function generate_batch(model::QwenModelCPU, tok, prompts::Vector{String}, state::GenerationState;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0)

    stop_tokens = Set{Int}()
    push!(stop_tokens, tok.eos_id)

    encode_fn = (p) -> getfield(parentmodule(typeof(tok)), :encode)(tok, p)
    decode_fn = (ids) -> getfield(parentmodule(typeof(tok)), :decode)(tok, ids)

    results = String[]
    for prompt in prompts
        prompt_tokens = encode_fn(prompt)
        if isempty(prompt_tokens)
            push!(results, "")
            continue
        end
        # Reset only the tracking state, not the KV cache buffers
        reset_state!(state, model)
        output = _generate_from_state(model, prompt_tokens, decode_fn, state;
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, stop_tokens=stop_tokens)
        push!(results, output)
    end

    return results
 end

 """
    stream_to_stdout_cpu(model, prompt_tokens, decode_fn; kwargs...)

Generate tokens and print them to stdout as they are produced.

Returns the complete generated text as a String.

MTP Support:
Set `use_mtp=true` to enable Multi-Token Prediction for faster generation.
"""
function stream_to_stdout_cpu(model::QwenModelCPU, prompt_tokens::Vector{Int}, decode_fn::Function;
 max_tokens::Int=100,
 temperature::Float32=0.7f0,
 top_p::Float32=0.95f0,
 top_k::Int=20,
 repetition_penalty::Float32=1.0f0,
 presence_penalty::Float32=0.0f0,
 min_p::Float32=0.0f0,
 stop_tokens::Set{Int}=Set{Int}(),
 show_tps::Bool=false,
 io::IO=stdout,
 max_context::Int=8192,
 # MTP options
 use_mtp::Bool=false,
 k_toks::Int=4,
 mask_id::Int=151643,
 # Interrupt support
interrupt_check::Function=() -> false)
   
  stream = generate_stream_cpu(model, prompt_tokens, decode_fn;
  max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, 
  repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, 
  min_p=min_p, stop_tokens=stop_tokens, max_context=max_context,
  use_mtp=use_mtp, k_toks=k_toks, mask_id=mask_id,
  interrupt_check=interrupt_check)
    
    generated_text = IOBuffer()
    try
        t0 = time()
        token_count = 0
        for token in stream
            if interrupt_check()
                break
            end
            print(io, token)
            flush(io)
            print(generated_text, token)
            token_count += 1
        end
        if token_count > 0
            println(io)
            flush(io)
        end
        if show_tps
            elapsed = time() - t0
            tps = elapsed > 0 ? token_count / elapsed : 0.0
            @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
            flush(io)
        end
        return String(take!(generated_text))
    catch e
        if e isa InterruptException
            println(io)
            flush(io)
            if show_tps
                elapsed = time() - t0
                tps = elapsed > 0 ? token_count / elapsed : 0.0
                @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
                flush(io)
            end
            return String(take!(generated_text))
        else
            rethrow(e)
        end
    end
end

"""
    stream_to_stdout_cpu(model, tok, prompt; kwargs...)

Generate from a string prompt and print tokens as they are produced.
This is a convenience method that handles tokenization internally.

# Example
```julia
model, file = load_model_cpu("model.gguf")
tok = SimpleTokenizer(file)
stream_to_stdout_cpu(model, tok, "Hello, how are you?")
```
"""
function stream_to_stdout_cpu(model::QwenModelCPU, tok, prompt::String; stop_tokens::Set{Int}=Set{Int}(), kwargs...)
 prompt_tokens = encode_prompt(tok, prompt)
 decode_fn = (ids) -> decode_tokens(tok, ids)
 # Add default stop tokens for Qwen3.5: EOS + <|im_end|>
 if isempty(stop_tokens) && hasfield(typeof(tok), :token_to_id)
 im_end_id = get(tok.token_to_id, "<|im_end|>", 0)
 eos_id = hasfield(typeof(tok), :eos_id) ? tok.eos_id : 0
 default_stops = Set(filter(!=(0), [eos_id, im_end_id]))
 if !isempty(default_stops)
 stop_tokens = default_stops
 end
 end
 return stream_to_stdout_cpu(model, prompt_tokens, decode_fn; stop_tokens=stop_tokens, kwargs...)
end

# Helper functions for tokenization
function encode_prompt(tok, prompt::String)
 # Handle Vector{String} (raw token list from GGUF)
 if tok isa Vector{String}
 tokens_data = tok
 tokens = Int[]
 remaining = prompt
 is_first = true
 while !isempty(remaining)
 found = false
 for len in length(remaining):-1:1
 candidate = SubString(remaining, 1, len)
 # For first token, try without prefix first, then with prefix
 # For subsequent tokens, try with prefix first (space prefix in BPE)
 prefixes = is_first ? ["", "Ġ"] : ["Ġ", ""]
 for prefix in prefixes
 key = prefix * candidate
 idx = findfirst(==(key), tokens_data)
 if idx !== nothing
 push!(tokens, idx) # 1-indexed for Julia arrays
 remaining = len < length(remaining) ? SubString(remaining, len + 1) : ""
 found = true
 is_first = false
 break
 end
 end
 found && break
 end
 if !found
 remaining = length(remaining) > 1 ? SubString(remaining, 2) : ""
 end
 end
 return tokens
    # Handle BPETokenizer (already 1-indexed, pass through)
    elseif hasfield(typeof(tok), :id_to_token) && hasfield(typeof(tok), :merges)
        return Base.invokelatest(getfield(parentmodule(typeof(tok)), :encode), tok, prompt)
    # Handle SimpleTokenizer struct (0-indexed -> convert to 1-indexed)
    elseif hasfield(typeof(tok), :token_to_id)
        tokens = Int[]
        remaining = prompt
        while !isempty(remaining)
            found = false
            for len in length(remaining):-1:1
                candidate = SubString(remaining, 1, len)
                for prefix in ["", "Ġ"]
                    key = prefix * candidate
                    if haskey(tok.token_to_id, key)
                        push!(tokens, tok.token_to_id[key] + 1)  # Convert 0-indexed to 1-indexed
                        remaining = len < length(remaining) ? SubString(remaining, len + 1) : ""
                        found = true
                        break
                    end
                end
                found && break
            end
            if !found
                remaining = length(remaining) > 1 ? SubString(remaining, 2) : ""
            end
        end
        return tokens
    # Handle BPETokenizer (1-indexed encode -> convert to 0-indexed)
    else
        # Fallback: assume tok is a function
        return tok(prompt)
    end
end

function decode_tokens(tok, ids::Vector{Int})
    # Handle Vector{String} (raw token list from GGUF)
    if tok isa Vector{String}
        parts = String[]
        for id in ids
            if 1 <= id <= length(tok)
                t = tok[id]
                t = replace(t, "Ġ" => " ")
                push!(parts, t)
            end
        end
        return join(parts)
    # Handle SimpleTokenizer struct (0-indexed ids -> need +1 for 1-indexed tokens vector)
    elseif hasfield(typeof(tok), :tokens)
        parts = String[]
        for id in ids
            if 1 <= id <= length(tok.tokens)
                t = tok.tokens[id]
                t = replace(t, "Ġ" => " ")
                push!(parts, t)
            end
        end
        return join(parts)
    # Handle BPETokenizer (already 1-indexed, pass through)
    elseif hasfield(typeof(tok), :id_to_token) && hasfield(typeof(tok), :merges)
        return Base.invokelatest(getfield(parentmodule(typeof(tok)), :decode), tok, ids)
 else
 return tok(ids)
 end
end

# --- Multi-Token Prediction (MTP) Functions ---

# NOTE: MTP requires a model specifically trained with Multi-Token Prediction objective.
# Standard Qwen3.5 models are NOT trained with MTP and will not work correctly.
# To use MTP, you need a model like "jwkirchenbauer/Qwen3-4B-Inst-2507-MTP".
#
# MTP models use a special mask token (typically 151669 for Qwen3) and are trained
# to predict multiple future tokens simultaneously. The mask token acts as a placeholder
# for positions where the model should predict the next token.

"""
 generate_mtp_cpu(model, prompt_tokens; k_toks=4, mask_id, kwargs...)

Generate tokens using Multi-Token Prediction (MTP) for speculative decoding.

**IMPORTANT**: This only works with models specifically trained for MTP.
Standard Qwen3.5 models are NOT trained with MTP and will produce incorrect output.

MTP allows predicting k future tokens in a single forward pass by appending
mask tokens to the input. The model outputs logits for all positions, and
the last k positions give predictions for the next k tokens.

Arguments:
- model: The QwenModelCPU model (must be MTP-trained)
- prompt_tokens: Initial prompt tokens
- k_toks: Number of tokens to predict per step (default: 4)
- mask_id: Token ID to use as mask (151669 for Qwen3-MTP models)
- max_tokens: Maximum tokens to generate
- temperature, top_p, top_k: Sampling parameters
- stop_tokens: Set of token IDs that stop generation
- strategy: :greedy or :adaptive (confidence-based)

Returns: Vector of generated token IDs

Example:
```julia
# For MTP-trained models only!
mask_id = 151669  # MTP mask token for Qwen3
tokens = generate_mtp_cpu(model, prompt_tokens; k_toks=4, mask_id=mask_id, max_tokens=100)
```

To use MTP, you need to:
1. Download an MTP-trained model (e.g., from huggingface)
2. Convert to GGUF or safetensors format
3. Load with the correct mask_id (check the model's README)
```
"""
function generate_mtp_cpu(model::QwenModelCPU, prompt_tokens::Vector{Int};
 k_toks::Int=4,
 mask_id::Int=151643, # Default to BOS, only used if no MTP head
 max_tokens::Int=512,
 temperature::Float32=0.0f0,  # 0 = greedy
 top_p::Float32=1.0f0,
 top_k::Int=0,
 stop_tokens::Set{Int}=Set{Int}(),
 strategy::Symbol=:greedy,  # :greedy, :adaptive
 confidence_threshold::Float32=0.9f0,
 max_context::Int=8192)
 
 # Initialize caches
 max_cache_seq = min(model.config.max_position_embeddings, max_context)
 caches = [init_kv_cache_cpu(model.config, max_cache_seq) for _ in 1:model.config.num_hidden_layers]
 reset_states_cpu!(model)
 
 generated_tokens = Int[]
 
 # Process prompt
 curr_pos = length(prompt_tokens)
 logits = forward_cpu!(model, prompt_tokens, 0, caches; full_logits=false)
 
 # Get first prediction from prompt
 first_logits = vec(logits)
 if temperature == 0.0f0
 next_token = argmax(first_logits)
 else
 next_token = softmax_sample(first_logits; temperature=temperature, top_p=top_p, top_k=top_k)
 end
 push!(generated_tokens, next_token)
 curr_pos += 1
 
 # Check if first token is stop
 if next_token in stop_tokens
 return generated_tokens
 end
 
 # MTP generation loop
 while length(generated_tokens) < max_tokens
 # Predict k tokens at once
 tokens_predicted = mtp_predict_step!(model, [next_token], curr_pos, caches;
 k_toks=k_toks, mask_id=mask_id,
 temperature=temperature, top_p=top_p, top_k=top_k,
 strategy=strategy, confidence_threshold=confidence_threshold)
 
 # Add predicted tokens
 for (i, tok) in enumerate(tokens_predicted)
 if tok in stop_tokens
 return generated_tokens
 end
 push!(generated_tokens, tok)
 if length(generated_tokens) >= max_tokens
 break
 end
 end
 
 # Update position
 curr_pos += length(tokens_predicted)
 
 # Last token becomes the seed for next prediction
 if !isempty(tokens_predicted)
 next_token = tokens_predicted[end]
 end
 end
 
 return generated_tokens
end

"""
 mtp_predict_step!(model, input_tokens, pos, caches; kwargs...)

Predict k future tokens in a single forward pass using MTP.

**With MTP head**: Uses the model's MTP head for prediction.
**Without MTP head**: Falls back to mask-token approach (requires MTP-trained model).

Returns a vector of k predicted tokens.
"""
function mtp_predict_step!(model::QwenModelCPU, input_tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU};
 k_toks::Int=4,
 mask_id::Int=151643, # Default to BOS token
 temperature::Float32=0.0f0,
 top_p::Float32=1.0f0,
 top_k::Int=0,
 strategy::Symbol=:greedy,
 confidence_threshold::Float32=0.9f0)
 
 if model.mtp !== nothing
 # Use actual MTP head for prediction
 return mtp_predict_with_head!(model, input_tokens, pos, caches;
 k_toks=k_toks, temperature=temperature,
 top_p=top_p, top_k=top_k, strategy=strategy,
 confidence_threshold=confidence_threshold)
 end
 
 # Fallback: mask-token approach (for models trained with mask tokens)
 # Build input with mask tokens: [token, MASK, MASK, ..., MASK]
 num_masks = k_toks - 1
 mtp_input = vcat(input_tokens, fill(mask_id, num_masks))
 
 # Forward pass with full logits to get predictions at all positions
 all_logits = forward_cpu!(model, mtp_input, pos, caches; full_logits=true)
 
 # Extract logits for the last k_toks positions
 last_k_logits = all_logits[:, end-k_toks+1:end]
 
 predicted_tokens = Int[]
 
 for i in 1:k_toks
 logits_i = last_k_logits[:, i]
 
 if strategy == :greedy || temperature == 0.0f0
 tok = argmax(logits_i)
 push!(predicted_tokens, tok)
 elseif strategy == :adaptive
 probs = softmax(logits_i)
 max_prob = maximum(probs)
 if max_prob < confidence_threshold && i > 1
 break
 end
 tok = argmax(probs)
 push!(predicted_tokens, tok)
 else
 tok = softmax_sample(logits_i; temperature=temperature, top_p=top_p, top_k=top_k)
 push!(predicted_tokens, tok)
 end
 end
 
 return predicted_tokens
end

"""
 mtp_predict_with_head!(model, input_tokens, pos, caches; kwargs...)

Use the MTP head to predict multiple future tokens.

The MTP head takes:
1. The last hidden state from the main model
2. The embedding of the predicted token

And produces logits for the next token. This is done iteratively to predict k tokens.
"""
function mtp_predict_with_head!(model::QwenModelCPU, input_tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU};
 k_toks::Int=4,
 temperature::Float32=0.0f0,
 top_p::Float32=1.0f0,
 top_k::Int=0,
 strategy::Symbol=:greedy,
 confidence_threshold::Float32=0.9f0)
 
 mtp = model.mtp
 @assert mtp !== nothing "MTP head is not loaded"
 
 predicted_tokens = Int[]
 
 # Get the last hidden state from the main model
 # Run forward pass through main model to get hidden state
 hidden = forward_hidden!(model, input_tokens, pos, caches)
 
 # Get first predicted token from main model
 first_logits = model.lm_head_buf
 first_token = argmax(first_logits)
 push!(predicted_tokens, first_token)
 
 # Now use MTP head to predict remaining tokens
 for i in 2:k_toks
 # Get embedding of last predicted token
 embed_i = model.embed[:, first_token]
 
 # MTP forward pass: norm(embed) + norm(hidden) -> fc -> logits
 mtp_logits = mtp_forward!(mtp, embed_i, hidden, model)
 
 if strategy == :greedy || temperature == 0.0f0
 next_token = argmax(mtp_logits)
 elseif strategy == :adaptive
 probs = softmax(mtp_logits)
 max_prob = maximum(probs)
 if max_prob < confidence_threshold
 break
 end
 next_token = argmax(probs)
 else
 next_token = softmax_sample(mtp_logits; temperature=temperature, top_p=top_p, top_k=top_k)
 end
 
 push!(predicted_tokens, next_token)
 first_token = next_token
 end
 
 return predicted_tokens
end

"""
 mtp_forward!(mtp, embedding, hidden, model)

Forward pass through the MTP head.

Takes the token embedding and hidden state, normalizes both,
combines them, and produces vocab logits for next token prediction.

The MTP fc projects from 2*hidden to hidden. The final prediction
uses the main model's lm_head to get vocab logits.
"""
function mtp_forward!(mtp::MTPHeadCPU, embedding::AbstractVector{Float32}, hidden::AbstractVector{Float32}, model::QwenModelCPU)
 # Normalize embedding
 rmsnorm_cpu!(mtp.embed_buf, embedding, mtp.pre_fc_norm_embedding)
 
 # Normalize hidden state
 rmsnorm_cpu!(mtp.hidden_buf, hidden, mtp.pre_fc_norm_hidden)
 
 # Combine: fc expects concatenated [norm_emb; norm_hidden]
 hidden_size = length(hidden)
 mtp.combined_buf[1:hidden_size] .= mtp.embed_buf
 mtp.combined_buf[hidden_size+1:end] .= mtp.hidden_buf
 
 # Project through fc: (hidden_size, 2*hidden_size) * (2*hidden_size,) -> (hidden_size,)
 mul!(mtp.fc_out_buf, mtp.fc, mtp.combined_buf)
 
 # Apply final norm
 rmsnorm_cpu!(mtp.fc_out_buf, mtp.fc_out_buf, mtp.norm)
 
 # Project to vocab using main model's lm_head: (vocab_size, hidden_size) * (hidden_size,) -> (vocab_size,)
 mul!(mtp.logits_buf, model.lm_head, mtp.fc_out_buf)
 
 return mtp.logits_buf
end

"""
 forward_hidden!(model, tokens, pos, caches)

Run forward pass and return the hidden state before LM head.
This is needed for MTP which takes the hidden state as input.
"""
function forward_hidden!(model::QwenModelCPU, tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU})
 seq_len = length(tokens)
 
 # For single token, just get embedding
 x = model.embed[:, tokens[end]]
 curr_pos = pos + seq_len - 1
 
 # Run through all layers
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 
 # Final norm (but don't project to vocab)
 rmsnorm_cpu!(model.final_norm_buf, x, model.final_norm)
 
 return model.final_norm_buf
end

"""
 softmax(logits)

Compute softmax of logits vector.
"""
function softmax(logits::AbstractVector{Float32})
 max_l = maximum(logits)
 exp_l = exp.(logits .- max_l)
 return exp_l ./ sum(exp_l)
end

"""
 mtp_verify_step!(model, draft_tokens, pos, caches; kwargs...)

Verify draft tokens using the model. Returns:
- (accepted_count, verification_logits)

This implements the verification step of speculative decoding where
the base model checks which draft tokens are correct.
"""
function mtp_verify_step!(model::QwenModelCPU, draft_tokens::Vector{Int}, pos::Int, caches::Vector{KVCacheCPU})
 # Run forward pass for all draft tokens
 logits = forward_cpu!(model, draft_tokens, pos, caches; full_logits=true)
 
 # For each position i, check if draft_tokens[i+1] matches argmax(logits[:, i])
 accepted = 0
 for i in 1:length(draft_tokens)-1
 predicted = argmax(logits[:, i])
 if predicted == draft_tokens[i+1]
 accepted += 1
 else
 break
 end
 end
 
 return accepted, logits
end

# Include Flash Attention and Speculative Decoding
include("FlashAttention.jl")
include("SpeculativeDecoding.jl")

# --- Model Display ---

function Base.show(io::IO, ::MIME"text/plain", model::QwenModelCPU)
    config = model.config
    arch = config.architecture

    println(io, "QwenModelCPU (CPU backend)")
    println(io, "├─ Architecture: ", arch)
    println(io, "├─ Hidden size: ", config.hidden_size)
    println(io, "├─ Layers: ", config.num_hidden_layers)
    println(io, "├─ Attention heads: ", config.num_attention_heads, " (KV: ", config.num_key_value_heads, ")")
    println(io, "├─ Head dim: ", config.head_dim)
    println(io, "├─ Intermediate size: ", config.intermediate_size)
    println(io, "├─ Vocab size: ", config.vocab_size)
    println(io, "├─ Max positions: ", config.max_position_embeddings)
    println(io, "├─ RoPE theta: ", config.rope_theta)
    println(io, "├─ Full attention interval: ", config.full_attention_interval)
    if hasfield(typeof(config), :partial_rotary_factor)
        println(io, "├─ Partial rotary factor: ", config.partial_rotary_factor)
    end

    # SSM params
    if hasfield(typeof(config), :ssm_inner_size) && config.ssm_inner_size > 0
        println(io, "├─ SSM inner size: ", config.ssm_inner_size)
        println(io, "├─ SSM state size: ", config.ssm_state_size)
        println(io, "├─ SSM group count: ", config.ssm_group_count)
        println(io, "├─ SSM time step rank: ", config.ssm_time_step_rank)
        println(io, "├─ SSM conv kernel: ", config.ssm_conv_kernel)
    end

    # MoE
    if hasfield(typeof(config), :num_experts) && config.num_experts > 0
        println(io, "├─ Experts: ", config.num_experts, " (top-k: ", config.num_experts_per_tok, ")")
    end

    # MLA (DeepSeek)
    if hasfield(typeof(config), :q_lora_rank) && config.q_lora_rank > 0
        println(io, "├─ MLA Q-LoRA rank: ", config.q_lora_rank)
        println(io, "├─ MLA KV-LoRA rank: ", config.kv_lora_rank)
        println(io, "├─ MLA QK RoPE head dim: ", config.qk_rope_head_dim)
        println(io, "├─ MLA QK NoPE head dim: ", config.qk_nope_head_dim)
        println(io, "├─ MLA V head dim: ", config.v_head_dim)
    end

    # MTP
    if model.mtp !== nothing
        println(io, "├─ MTP head: enabled")
    end

    # Precision
    bf16_status = config.use_bf16_weights ? "enabled" : "disabled"
    println(io, "└─ BF16 weights: ", bf16_status)
end

function Base.show(io::IO, model::QwenModelCPU)
    show(io, MIME"text/plain"(), model)
end

end # module
