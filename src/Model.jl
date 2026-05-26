module Model

using LinearAlgebra
using Statistics
# GPU backend (optional)
try
    using oneAPI
    using oneAPI.oneMKL
catch
end

export QwenConfig, QwenModel, KVCache, forward!, RMSNorm, MLP, MoE, GatedDeltaNet, FullAttention, MLAttention, DecoderLayer, init_kv_cache, free_kv_cache!, free_all_kv_caches!, free_model_gpu!

# --- GPU Array Aliases ---
const oneArray = oneAPI.oneArray
const oneVector{T} = oneArray{T,1}
const oneMatrix{T} = oneArray{T,2}

# Abstract types for function signatures
const AbstractVectorH{T} = AbstractVector{T}
const AbstractMatrixH{T} = AbstractMatrix{T}
const AbstractArrayH{T,N} = AbstractArray{T,N}
const AbstractArrayH{T} = AbstractArray{T}
const AbstractArrayH = AbstractArray

# --- Configuration ---
Base.@kwdef struct QwenConfig
 architecture::Symbol = :qwen
 vocab_size::Int = 151936
 hidden_size::Int = 1024
 intermediate_size::Int = 3584
 num_hidden_layers::Int = 24
 num_attention_heads::Int = 8 # q heads for full-attention layers
 num_key_value_heads::Int = 2 # kv heads for full-attention layers
 head_dim::Int = 256 # from attn_key_length
 rms_norm_eps::Float16 = Float16(1e-6)
 rope_theta::Float32 = 10000000.0f0 # Use Float32 to avoid overflow
 max_position_embeddings::Int = 4096
 full_attention_interval::Int = 4
 ssm_inner_size::Int = 2048
 ssm_state_size::Int = 128 # head_k_dim
 ssm_group_count::Int = 16 # num_k_heads = num_v_heads
 ssm_time_step_rank::Int = 16 # num_v_heads
    ssm_conv_kernel::Int = 4
    # MoE
    num_experts::Int = 0
    num_experts_per_tok::Int = 0
    # MLA (Multi-Head Latent Attention for DeepSeek)
    q_lora_rank::Int = 0
    kv_lora_rank::Int = 0
    qk_rope_head_dim::Int = 0
    qk_nope_head_dim::Int = 0  # Non-RoPE head dimension for Q/K
    v_head_dim::Int = 0
end

abstract type QuantMatrix end
struct IQ2XXSMatrix <: QuantMatrix end

function free_gpu_tables!()
end

function init_gpu_tables(grid, signs_table, kmask)
end

# --- Normalization ---
struct RMSNorm{W<:AbstractVector}
    weight::W
    eps::Float16
end

function (norm::RMSNorm)(x::AbstractArray)
    T = eltype(x)
    # Use mapreduce for stable reduction. Upcast to Float32 during reduction to prevent overflow.
    # ss will be size (1, batch...)
    ss = mapreduce(v -> Float32(abs2(v)), +, x, dims=1)

    m = ss ./ Float32(size(x, 1))
    scale = @. T(1.0f0 / sqrt(m + Float32(norm.eps)))

    return @. x * scale * norm.weight
end

function rmsnorm!(out::AbstractArray, x::AbstractArray, norm::RMSNorm)
 T = eltype(x)
 ss = mapreduce(v -> Float32(abs2(v)), +, x, dims=1)
 m = ss ./ Float32(size(x, 1))
 scale = @. T(1.0f0 / sqrt(m + Float32(norm.eps)))
 out .= x .* scale .* norm.weight
 return out
end

# CPU-only version that takes weight vector directly (avoids GPU scalar indexing)
function rmsnorm_cpu!(out::AbstractArray{Float32}, x::AbstractArray{Float32}, weight::AbstractVector{Float32}, eps::Float32)
 ss = mapreduce(v -> Float32(abs2(v)), +, x, dims=1)
 m = ss ./ Float32(size(x, 1))
 scale = @. Float32(1.0f0 / sqrt(m + eps))
 out .= x .* scale .* weight
 return out
end

function rmsnorm(x::AbstractArray, norm::RMSNorm)
    return norm(x)
end

# --- GPU Sampling Kernels ---

# Temperature scaling kernel
function temperature_scale_kernel!(scaled_logits, logits, inv_temp, N)
    i = get_global_id(1)
    if i <= N
        scaled_logits[i] = logits[i] * inv_temp
    end
    return
end

function mat_mul!(res::AbstractMatrix, weight::AbstractMatrix, x::AbstractMatrix)
    mul!(res, weight, x)
    return res
end

function mat_mul(weight::AbstractMatrix, x::AbstractMatrix)
    return weight * x
end

function mat_mul_AT_B!(res::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
 # Use oneMKL.gemm! directly with 'T' for transpose
 oneAPI.oneMKL.gemm!('T', 'N', one(Float16), A, B, zero(Float16), res)
 return res
end

function mat_mul_AT_B(A::AbstractMatrix, B::AbstractMatrix)
 # For non-mutating version, create result array and call gemm!
 res = similar(B, (size(A, 2), size(B, 2)))
 oneAPI.oneMKL.gemm!('T', 'N', one(Float16), A, B, zero(Float16), res)
 return res
end

function mat_mul_AB(A::AbstractMatrix, B::AbstractMatrix)
    return A * B
end


# --- GPU Softmax Kernel ---
# Stable softmax on a 1D slice of length `len` stored in `scores[1:len, 1]`
# Writes probabilities back into `probs[1:len, 1]`
# Uses Float32 for exp() to prevent overflow
function softmax_kernel!(probs, scores, len, scale)
 # Copy to CPU for computation (avoids scalar indexing on GPU)
 probs_cpu = Array(probs)
 scores_cpu = Array(scores)

 # Compute max in Float32 for stability
 mx = Float32(maximum(scores_cpu[1:len, 1]) * scale)
 s = Float32(0.0)
 @inbounds for i in 1:len
 v = exp(Float32(scores_cpu[i, 1] * scale) - mx)
 probs_cpu[i, 1] = v
 s += v
 end
 inv_s = Float32(1.0) / s
 @inbounds for i in 1:len
 probs_cpu[i, 1] *= inv_s
 end

 # Copy back to GPU
 copyto!(probs, Float16.(probs_cpu))
 return nothing
end

function batched_softmax_kernel!(probs, scores, total_len, n_heads, scale)
 # Copy to CPU for computation (avoids scalar indexing on GPU)
 probs_cpu = Array(probs)
 scores_cpu = Array(scores)

 for h in 1:n_heads
 mx = Float32(maximum(scores_cpu[1:total_len, h]) * scale)
 s = Float32(0.0)
 @inbounds for i in 1:total_len
 v = exp(Float32(scores_cpu[i, h] * scale) - mx)
 probs_cpu[i, h] = v
 s += v
 end
 inv_s = Float32(1.0) / s
 @inbounds for i in 1:total_len
 probs_cpu[i, h] *= inv_s
 end
 end

 # Copy back to GPU
 copyto!(probs, Float16.(probs_cpu))
 return nothing
end

# Numerically stable sigmoid using Float32 intermediate computation
# Float16 exp() overflows/underflows for |x| > 6.5
function sigmoid_kernel!(out, x, N)
 for i in 1:N
 # Use Float32 for exp() to avoid overflow/underflow
 x_f32 = Float32(x[i])
 sigmoid_f32 = 1.0f0 / (1.0f0 + exp(-x_f32))
 out[i] = Float16(sigmoid_f32)
 end
 return nothing
end

# Numerically stable softplus using Float32 intermediate computation
function softplus_kernel!(out, x, bias, N)
 for i in 1:N
 # Use Float32 for exp() to avoid overflow/underflow
 x_f32 = Float32(x[i] + bias[i])
 # Use softplus identity: log(1 + exp(x)) ≈ max(x, 0) + log(1 + exp(-|x|))
 # This is numerically stable for all x
 if x_f32 > 20.0f0
 # For large x, log(1 + exp(x)) ≈ x
 softplus_f32 = x_f32
 elseif x_f32 < -20.0f0
 # For large negative x, log(1 + exp(x)) ≈ 0
 softplus_f32 = 0.0f0
 else
 softplus_f32 = log(1.0f0 + exp(x_f32))
 end
 out[i] = Float16(softplus_f32)
 end
 return nothing
end

# Numerically stable SiLU (Swish) using Float32 intermediate computation
# SiLU(x) = x * sigmoid(x)
function silu_kernel!(out, x, N)
 for i in 1:N
 x_f32 = Float32(x[i])
 sigmoid_f32 = 1.0f0 / (1.0f0 + exp(-x_f32))
 silu_f32 = x_f32 * sigmoid_f32
 out[i] = Float16(silu_f32)
 end
 return nothing
end

# Numerically stable SiLU gating with Float32 intermediate computation
# SiLU_gating(z, normed) = z * sigmoid(z) * normed
function silu_gating_kernel!(out, z, normed, N)
 for i in 1:N
 z_f32 = Float32(z[i])
 sigmoid_f32 = 1.0f0 / (1.0f0 + exp(-z_f32))
 silu_f32 = z_f32 * sigmoid_f32 * Float32(normed[i])
 out[i] = Float16(silu_f32)
 end
 return nothing
end

# --- SiLU ---
function silu(x::AbstractArray)
    T = eltype(x)
    return x .* (T(1.0) ./ (T(1.0) .+ exp.(-x)))
end

# --- Rotary Embedding (RoPE) ---
# --- Rotary Embedding (RoPE) ---
struct RotaryEmbedding
 dim::Int
 base::Float32  # Use Float32 for base to prevent precision loss
 inv_freq::Vector{Float32}  # Use Float32 for more precision
 rotary_dim::Int  # Number of dimensions that get rotary (partial rotary support)
end

function RotaryEmbedding(dim::Int; base=Float32(10000000.0), rotary_dim::Int=dim)
 # Only compute inv_freq for the rotary dimensions
 inv_freq = Float32(1.0) ./ (Float32(base) .^ (Float32.(range(0, stop=rotary_dim - 1, step=2)) ./ Float32(rotary_dim)))
 return RotaryEmbedding(dim, Float32(base), inv_freq, rotary_dim)
end

function rope_kernel!(x, inv_freq, pos, d, h, seq, d_rope)
 T = eltype(x)
 half_d = d_rope ÷ 2

 # Copy to CPU, compute, copy back (avoids scalar indexing on GPU)
 x_cpu = Array(x)
 inv_freq_cpu = Array(inv_freq)

 for t in 1:seq
 for head in 1:h
 for i in 1:half_d
 # Half-split pairing: first half paired with second half
 idx1 = i
 idx2 = i + half_d

 # Use Float32 for position computation to prevent precision loss
 p = Float32(pos + t - 1)
 freq = inv_freq_cpu[i] * p  # inv_freq is Float32
 sin_val, cos_val = sincos(freq)

 x1 = x_cpu[idx1, head, t]
 x2 = x_cpu[idx2, head, t]

 x_cpu[idx1, head, t] = x1 * cos_val - x2 * sin_val
 x_cpu[idx2, head, t] = x1 * sin_val + x2 * cos_val
 end
 end
 end

 copyto!(x, x_cpu)
 return x
end

function (rope::RotaryEmbedding)(x::AbstractArray{T,3}, pos::Int) where T
    d, h, seq = size(x)
    d_rope = min(d, rope.rotary_dim)
    rope_kernel!(x, rope.inv_freq, pos, d, h, seq, d_rope)
    return x
end

# --- KV Cache ---
mutable struct KVCache{T<:AbstractArray,V<:AbstractVector,M<:AbstractMatrix}
    k::T # (head_dim, n_kv, max_seq)
    v::T
    pos::Int

 # --- Scratchpad Buffers (To avoid any memory allocations) ---
 # Attention buffers
 q_all::V # Larger buffer for both Q and Gate (n_heads_q * head_dim * 2)
 k_buf::V # (n_heads_kv * head_dim)
 v_buf::V # (n_heads_kv * head_dim)
 scores::V # (max_pos,)
 attn_out_buf::V # (n_heads_q * head_dim)
 # MLP buffers (2D column matrices for GPU matmul)
 mlp_gate::M # (intermediate_size, 1)
 mlp_up::M # (intermediate_size, 1)
 branch_out::M # (hidden_size, 1)
 # RoPE buffers
 rope_q_tmp::V # (rope_pairs, n_heads_q)
 rope_k_tmp::V # (rope_pairs, n_heads_kv)
 # Normalization buffers
 norm1_buf::V # (hidden_size,)
 norm2_buf::V # (hidden_size,)
end

# Free GPU memory for a KVCache - call this on error/cleanup
function free_kv_cache!(cache::KVCache)
    try
        if isdefined(cache, :k) && cache.k !== nothing
            cache.k = oneArray{Float16,3}(undef, 0, 0, 0)  # Replace with empty array
        end
    catch e
        @warn "Error freeing KV cache k" exception = (e, catch_backtrace())
    end
    try
        if isdefined(cache, :v) && cache.v !== nothing
            cache.v = oneArray{Float16,3}(undef, 0, 0, 0)
        end
    catch e
        @warn "Error freeing KV cache v" exception = (e, catch_backtrace())
    end
    cache.pos = 0
    return nothing
end

# Free all KV caches in a vector
function free_all_kv_caches!(caches::Vector{KVCache})
    for cache in caches
        free_kv_cache!(cache)
    end
    return nothing
end

function init_kv_cache(config::QwenConfig)
 head_dim = config.head_dim
 n_heads_q = config.num_attention_heads
 n_heads_kv = config.num_key_value_heads
 max_pos = config.max_position_embeddings
 intermediate_size = config.intermediate_size
 hidden_size = config.hidden_size
 rope_pairs = head_dim ÷ 2

 try
 # K and V caches
 k = oneArray{Float16}(undef, head_dim, n_heads_kv, max_pos)
 v = oneArray{Float16}(undef, head_dim, n_heads_kv, max_pos)
 fill!(k, Float16(0.0))
 fill!(v, Float16(0.0))
 oneAPI.synchronize() # Ensure cache is zeroed before use

 # Attention buffers
 q_all = oneArray(zeros(Float16, n_heads_q * head_dim * 2))
 k_buf = oneArray(zeros(Float16, n_heads_kv * head_dim))
 v_buf = oneArray(zeros(Float16, n_heads_kv * head_dim))
 scores = oneArray(zeros(Float16, max_pos))
 attn_out_buf = oneArray(zeros(Float16, n_heads_q * head_dim))

 # MLP buffers (2D column matrices for GPU matmul)
 mlp_gate = oneArray(zeros(Float16, intermediate_size, 1))
 mlp_up = oneArray(zeros(Float16, intermediate_size, 1))
 branch_out = oneArray(zeros(Float16, hidden_size, 1))

 # RoPE buffers
 rope_q_tmp = oneArray(zeros(Float16, rope_pairs * n_heads_q))
 rope_k_tmp = oneArray(zeros(Float16, rope_pairs * n_heads_kv))

 # Normalization buffers
 norm1_buf = oneArray(zeros(Float16, hidden_size))
 norm2_buf = oneArray(zeros(Float16, hidden_size))

 return KVCache(k, v, 0, q_all, k_buf, v_buf, scores, attn_out_buf,
 mlp_gate, mlp_up, branch_out,
 rope_q_tmp, rope_k_tmp, norm1_buf, norm2_buf)
 catch e
 @error "Failed to initialize KV cache" exception = (e, catch_backtrace())
 rethrow(e)
 end
end

function update_kv_cache!(cache::KVCache, k::AbstractArray{Float16,2}, v::AbstractArray{Float16,2}, pos::Int)
 # pos is 0-indexed, convert to 1-indexed for Julia arrays
 @views cache.k[:, :, pos + 1] .= k
 @views cache.v[:, :, pos + 1] .= v
 cache.pos = pos + 1
 return cache.k, cache.v
end

function update_kv_cache!(cache::KVCache, k::AbstractArray{Float16,3}, v::AbstractArray{Float16,3}, pos::Int)
 # For 3D arrays (prefill), store each position in the sequence
 # pos is 0-indexed, convert to 1-indexed for Julia arrays
 hd, n_kv, seq = size(k)
 for t in 1:seq
 @views cache.k[:, :, pos + t] .= k[:, :, t]
 @views cache.v[:, :, pos + t] .= v[:, :, t]
 end
 cache.pos = pos + seq
 return cache.k, cache.v
end

# --- MLP ---

struct MLP
 index::Int # Layer index
 gate_weight::Union{oneMatrix{Float16},QuantMatrix}
 up_weight::Union{oneMatrix{Float16},QuantMatrix}
 down_weight::Union{oneMatrix{Float16},QuantMatrix}
end

function MLP(index::Int, gate_weight, up_weight, down_weight)
 return MLP(index, gate_weight, up_weight, down_weight)
end

function (m::MLP)(x::oneMatrix{Float16}, cache::KVCache)
 # GPU path - use numerically stable SiLU with Float32 intermediate
 mul!(cache.mlp_gate, m.gate_weight, x)
 mul!(cache.mlp_up, m.up_weight, x)
 
 # Numerically stable SiLU: gate * sigmoid(gate)
 # Use Float32 for exp() to avoid overflow/underflow
 gate_cpu = Array(cache.mlp_gate)
 up_cpu = Array(cache.mlp_up)
 
 # Apply SiLU in Float32 for numerical stability
 @. gate_cpu = gate_cpu * (Float32(1.0) / (Float32(1.0) + exp(-Float32(gate_cpu))))
 gate_cpu .*= up_cpu
 
 # Copy back to GPU
 copyto!(cache.mlp_gate, Float16.(gate_cpu))
 mul!(cache.branch_out, m.down_weight, cache.mlp_gate)

 return cache.branch_out
end

function (m::MLP)(x::oneMatrix{Float16})
 # Numerically stable SiLU using Float32 intermediate
 g = Array(mat_mul(m.gate_weight, x))
 u = Array(mat_mul(m.up_weight, x))
 # SiLU: g * sigmoid(g) in Float32
 @. g = g * (Float32(1.0) / (Float32(1.0) + exp(-Float32(g))))
 g .*= u
 return mat_mul(m.down_weight, oneArray(Float16.(g)))
end

function reset_states!(m::MLP)
 # MLP has no state to reset
 return nothing
end

# --- MoE (Mixture of Experts) ---

struct MoE
 index::Int # Layer index
 gate::Union{oneMatrix{Float16},QuantMatrix}
 experts_gate::Vector{Union{oneMatrix{Float16},QuantMatrix}}
 experts_up::Vector{Union{oneMatrix{Float16},QuantMatrix}}
 experts_down::Vector{Union{oneMatrix{Float16},QuantMatrix}}
 num_experts::Int
 num_experts_per_tok::Int
end

function MoE(index::Int, gate, experts_gate, experts_up, experts_down, num_experts, num_experts_per_tok)
 return MoE(index, gate, experts_gate, experts_up, experts_down, num_experts, num_experts_per_tok)
end

function (m::MoE)(x::oneMatrix{Float16}, cache::KVCache)
 # x: (hidden_size, seq_len)
 hidden_size, seq_len = size(x)

 # GPU path
 gate_logits = mat_mul(m.gate, x)
 gate_logits_cpu = collect(gate_logits)

 output = cache.branch_out
 fill!(output, Float16(0.0))

 for t in 1:seq_len
 logits = gate_logits_cpu[:, t]
 exp_logits = exp.(logits .- maximum(logits))
 probs = exp_logits ./ sum(exp_logits)

 top_k_indices = sortperm(probs, rev=true)[1:m.num_experts_per_tok]
 top_k_probs = probs[top_k_indices]
 top_k_probs ./= sum(top_k_probs)

 xt = @view x[:, t:t]

 for (i, expert_idx) in enumerate(top_k_indices)
 g = Array(mat_mul(m.experts_gate[expert_idx], xt))
 u = Array(mat_mul(m.experts_up[expert_idx], xt))
 # Numerically stable SiLU in Float32
 @. g = g * (Float32(1.0) / (Float32(1.0) + exp(-Float32(g))))
 g .*= u
 res = mat_mul(m.experts_down[expert_idx], oneArray(Float16.(g)))

 prob = top_k_probs[i]
 out_view = @view output[:, t:t]
 @. out_view += prob * res
 end
 end

 return output
end

function (m::MoE)(x::oneMatrix{Float16})
 # Numerically stable MoE using Float32 intermediate for SiLU
 hidden_size, seq_len = size(x)
 gate_logits = mat_mul(m.gate, x)
 gate_logits_cpu = collect(gate_logits)
 output = oneArray(zeros(Float16, hidden_size, seq_len))

 for t in 1:seq_len
 logits = gate_logits_cpu[:, t]
 exp_logits = exp.(logits .- maximum(logits))
 probs = exp_logits ./ sum(exp_logits)
 top_k_indices = sortperm(probs, rev=true)[1:m.num_experts_per_tok]
 top_k_probs = probs[top_k_indices]
 top_k_probs ./= sum(top_k_probs)

 xt = @view x[:, t:t]
 for (i, expert_idx) in enumerate(top_k_indices)
 g = Array(mat_mul(m.experts_gate[expert_idx], xt))
 u = Array(mat_mul(m.experts_up[expert_idx], xt))
 # Numerically stable SiLU in Float32
 @. g = g * (Float32(1.0) / (Float32(1.0) + exp(-Float32(g))))
 g .*= u
 res = mat_mul(m.experts_down[expert_idx], oneArray(Float16.(g)))
 prob = top_k_probs[i]
 out_view = @view output[:, t:t]
 @. out_view += prob * res
 end
 end

 return output
end

function reset_states!(m::MoE)
    # MoE has no state to reset
    return nothing
end

# --- Full Attention Layer ---
# Supports Qwen (gated Q), Llama (standard), Phi3 (packed QKV)

struct FullAttention
 index::Int # Layer index
 architecture::Symbol
 wq::Union{oneMatrix{Float16},QuantMatrix,Nothing}
 wk::Union{oneMatrix{Float16},QuantMatrix,Nothing}
 wv::Union{oneMatrix{Float16},QuantMatrix,Nothing}
 wqkv::Union{oneMatrix{Float16},QuantMatrix,Nothing} # For Phi3
 wo::Union{oneMatrix{Float16},QuantMatrix}
 q_norm::Union{RMSNorm,Nothing}
 k_norm::Union{RMSNorm,Nothing}
 n_heads::Int
 n_kv::Int
 head_dim::Int

 # Pre-allocated GPU buffers
 decode_q_full::oneMatrix{Float16}
 decode_k::oneMatrix{Float16}
 decode_v::oneMatrix{Float16}
 decode_combined::oneMatrix{Float16}
 decode_scores::oneMatrix{Float16}
 decode_pb::oneMatrix{Float16}
 decode_out_h::oneMatrix{Float16}
 decode_wo_buf::oneMatrix{Float16}
 # GPU buffers for prefill (dynamically sized)
 prefill_scores::oneMatrix{Float16}
 prefill_pb::oneMatrix{Float16}
end

function FullAttention(index::Int, arch, wq, wk, wv, wqkv, wo, q_norm, k_norm, n_heads, n_kv, hd, config::QwenConfig)
 q_size = (arch == :qwen || arch == :qwen2 || arch == :qwen2_5 || arch == :qwen35) ? hd * 2 * n_heads : hd * n_heads
 decode_q_full = oneArray(zeros(Float16, q_size, 1))
 decode_k = oneArray(zeros(Float16, hd * n_kv, 1))
 decode_v = oneArray(zeros(Float16, hd * n_kv, 1))
 decode_combined = oneArray(zeros(Float16, hd * n_heads, 1))

 # Fixed buffer sizes - sufficient for 4096 context
 max_len = 4096
 decode_scores = oneArray(zeros(Float16, max_len, 1))
 decode_pb = oneArray(zeros(Float16, max_len, 1))
 decode_out_h = oneArray(zeros(Float16, hd, 1))
 wo_out_size = size(wo, 1)
 decode_wo_buf = oneArray(zeros(Float16, wo_out_size, 1))

 # Prefill buffers - fixed size for 4096 context
 prefill_scores = oneArray(zeros(Float16, max_len, n_heads))
 prefill_pb = oneArray(zeros(Float16, max_len, n_heads))

 return FullAttention(index, arch, wq, wk, wv, wqkv, wo, q_norm, k_norm, n_heads, n_kv, hd,
 decode_q_full, decode_k, decode_v, decode_combined, decode_scores, decode_pb, decode_out_h,
 decode_wo_buf, prefill_scores, prefill_pb)
end

function reset_states!(m::FullAttention)
 # FullAttention has no state to reset
 return nothing
end

function (m::FullAttention)(x::oneArray{Float16,2}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    hd, seq = m.head_dim, size(x, 2)
    T = Float16
    n_heads_q = m.n_heads
    n_heads_kv = m.n_kv
    gqa_ratio = div(n_heads_q, n_heads_kv)

    if m.architecture == :qwen || m.architecture == :qwen2 || m.architecture == :qwen2_5 || m.architecture == :qwen35
        # 1. Packed Q+gate projection
        if seq == 1
            q_full = mat_mul!(m.decode_q_full, m.wq, x)
            q_3d = reshape(q_full, div(size(q_full, 1), m.n_heads), m.n_heads, seq)
            q_only = view(q_3d, 1:hd, :, :)
            gate_raw = view(q_3d, hd+1:2*hd, :, :)
            k = mat_mul!(m.decode_k, m.wk, x)
            v = mat_mul!(m.decode_v, m.wv, x)
        else
            q_full = mat_mul(m.wq, x)
            q_3d = reshape(q_full, div(size(q_full, 1), m.n_heads), m.n_heads, seq)
            q_only = view(q_3d, 1:hd, :, :)
            gate_raw = view(q_3d, hd+1:2*hd, :, :)
            k = mat_mul(m.wk, x)
            v = mat_mul(m.wv, x)
        end

        # 2. Apply Q, K normalization
        q_normed = m.q_norm(reshape(q_only, hd, :))
        k_normed = m.k_norm(reshape(k, hd, :))
        q_2d = reshape(q_normed, hd, m.n_heads, seq)
        k_2d = reshape(k_normed, hd, m.n_kv, seq)
        v_2d = reshape(v, hd, m.n_kv, seq)

        # 3. RoPE
        q_rope = rope(q_2d, pos)
        k_rope = rope(k_2d, pos)

 # 4. Gate (applied AFTER attention, not to Q)
 gate_raw_cpu = Array(gate_raw)
 @. gate_raw_cpu = Float32(1.0) / (Float32(1.0) + exp(-Float32(gate_raw_cpu))) # sigmoid
 gate_sigmoid = oneArray(Float16.(gate_raw_cpu))
 q_gated = q_rope

    elseif m.architecture == :phi3
        # Packed QKV
        qkv = mat_mul(m.wqkv, x)
        q_flat = view(qkv, 1:hd*m.n_heads, :)
        k_flat = view(qkv, hd*m.n_heads+1:hd*(m.n_heads+m.n_kv), :)
        v_flat = view(qkv, hd*(m.n_heads+m.n_kv)+1:hd*(m.n_heads+2*m.n_kv), :)

        q_2d = reshape(q_flat, hd, m.n_heads, seq)
        k_2d = reshape(k_flat, hd, m.n_kv, seq)
        v_2d = reshape(v_flat, hd, m.n_kv, seq)

        q_rope = rope(q_2d, pos)
        k_rope = rope(k_2d, pos)
        q_gated = q_rope
    else
        # Standard Llama/Mistral/GQA
        if seq == 1
            q_flat = mat_mul!(m.decode_q_full, m.wq, x)
            k_flat = mat_mul!(m.decode_k, m.wk, x)
            v_flat = mat_mul!(m.decode_v, m.wv, x)
        else
            q_flat = mat_mul(m.wq, x)
            k_flat = mat_mul(m.wk, x)
            v_flat = mat_mul(m.wv, x)
        end

        q_2d = reshape(q_flat, hd, m.n_heads, seq)
        k_2d = reshape(k_flat, hd, m.n_kv, seq)
        v_2d = reshape(v_flat, hd, m.n_kv, seq)

        q_rope = rope(q_2d, pos)
        k_rope = rope(k_2d, pos)
        q_gated = q_rope
    end

 # 5. KV Cache update
 K_cache, V_cache = update_kv_cache!(cache, k_rope, v_2d, pos)

 # 6. Attention
 total_len = cache.pos
 # Use Float32 for scale to prevent precision loss
 scale = Float32(1.0) / sqrt(Float32(hd))

    if seq == 1
        # Decode path
        q_final = reshape(q_gated, hd, m.n_heads, 1)

        for h in 1:m.n_heads
            kh = div(h - 1, gqa_ratio) + 1

            sc_view = view(m.decode_scores, 1:total_len, :)
            K_view = view(K_cache, :, kh, 1:total_len)
            q_view = view(q_final, :, h, :)

            mat_mul_AT_B!(sc_view, K_view, q_view)
            sc_view .*= scale

            pb_view = view(m.decode_pb, 1:total_len, :)
            softmax_kernel!(pb_view, sc_view, total_len, T(1.0))

            out_view = view(m.decode_combined, (h-1)*hd+1:h*hd, :)
            V_view = view(V_cache, :, kh, 1:total_len)
            mul!(out_view, V_view, pb_view)
        end

        combined = m.decode_combined
        combined .*= gate_sigmoid
        result = mat_mul!(m.decode_wo_buf, m.wo, combined)

 else
 # Prefill path
 q_final = reshape(q_gated, hd, m.n_heads, seq)
 combined_all = zeros(T, hd * m.n_heads, seq)

 for h in 1:m.n_heads
 kh = div(h - 1, gqa_ratio) + 1
 for s in 1:seq
 q_v = @view q_final[:, h, s]
 K_v = @view K_cache[:, kh, 1:(pos+s)]
 V_v = @view V_cache[:, kh, 1:(pos+s)]

 # Use Float32 for attention scores to prevent overflow
 scores = zeros(Float32, pos + s)
 for i in 1:(pos+s)
 dot = Float32(0.0)
 for d in 1:hd
 dot += Float32(K_v[d, i]) * Float32(q_v[d])
 end
 scores[i] = dot * scale
 end

 # Stable softmax in Float32
 mx = maximum(scores)
 sum_exp = Float32(0.0)
 for i in 1:(pos+s)
 scores[i] = exp(scores[i] - mx)
 sum_exp += scores[i]
 end
 scores ./= sum_exp

 for d in 1:hd
 val = T(0.0)
 for i in 1:(pos+s)
 val += V_v[d, i] * T(scores[i])
 end
 combined_all[(h-1)*hd+d, s] = val
 end
 end
 end
 combined = oneArray(combined_all)
 combined .*= gate_sigmoid
 result = mat_mul(m.wo, combined)
 end

 return result
end

# --- MLAttention (Multi-Latent Attention for DeepSeek V2/V3) ---
struct MLAttention
    q_a_proj::Union{oneMatrix{Float16},QuantMatrix}
    q_a_norm::RMSNorm
    q_b_proj::Union{oneMatrix{Float16},QuantMatrix}
    kv_a_proj_with_mqa::Union{oneMatrix{Float16},QuantMatrix}
    kv_a_norm::RMSNorm
    kv_b_proj::Union{oneMatrix{Float16},QuantMatrix}
    wo::Union{oneMatrix{Float16},QuantMatrix}

    n_heads::Int
    head_dim::Int
    q_lora_rank::Int
    kv_lora_rank::Int
    qk_rope_head_dim::Int
    v_head_dim::Int
    qk_nope_head_dim::Int  # Non-RoPE dimension per head for Q/K
    softmax_scale::Float16

    # Pre-allocated GPU buffers for decode (single token)
    decode_q_latent::oneMatrix{Float16}      # (q_lora_rank, 1)
    decode_q_latent_norm::oneMatrix{Float16} # (q_lora_rank, 1)
    decode_q_all::oneMatrix{Float16}         # (n_heads * (qk_nope + qk_rope), 1)
    decode_kv_a::oneMatrix{Float16}          # (kv_lora_rank + qk_rope_head_dim, 1)
    decode_kv_latent::oneMatrix{Float16}     # (kv_lora_rank, 1)
    decode_kv_latent_norm::oneMatrix{Float16}# (kv_lora_rank, 1)
    decode_kv_b::oneMatrix{Float16}          # (n_heads * (qk_nope + v_head_dim), 1)
    decode_scores::oneMatrix{Float16}        # (max_seq, n_heads)
    decode_pb::oneMatrix{Float16}            # (max_seq, n_heads)
    decode_out::oneMatrix{Float16}           # (n_heads * v_head_dim, 1)
    decode_wo_buf::oneMatrix{Float16}        # (hidden_size, 1)

    # CPU scratchpads for numerical stability
    q_pe_cpu::Vector{Float32}
    k_pe_cpu::Vector{Float32}
    scores_cpu::Vector{Float32}
    pb_cpu::Vector{Float32}
end

function MLAttention(q_a_proj, q_a_norm, q_b_proj, kv_a_proj_with_mqa, kv_a_norm, kv_b_proj, wo,
    n_heads::Int, head_dim::Int, q_lora_rank::Int, kv_lora_rank::Int,
    qk_rope_head_dim::Int, v_head_dim::Int, qk_nope_head_dim::Int,
    softmax_scale::Float16, config::QwenConfig)

    max_seq = config.max_position_embeddings

    # Pre-allocated GPU buffers for decode path
    decode_q_latent = oneArray(zeros(Float16, q_lora_rank, 1))
    decode_q_latent_norm = oneArray(zeros(Float16, q_lora_rank, 1))
    decode_q_all = oneArray(zeros(Float16, n_heads * (qk_nope_head_dim + qk_rope_head_dim), 1))
    decode_kv_a = oneArray(zeros(Float16, kv_lora_rank + qk_rope_head_dim, 1))
    decode_kv_latent = oneArray(zeros(Float16, kv_lora_rank, 1))
    decode_kv_latent_norm = oneArray(zeros(Float16, kv_lora_rank, 1))
    decode_kv_b = oneArray(zeros(Float16, n_heads * (qk_nope_head_dim + v_head_dim), 1))
    decode_scores = oneArray(zeros(Float16, max_seq, n_heads))
    decode_pb = oneArray(zeros(Float16, max_seq, n_heads))
    decode_out = oneArray(zeros(Float16, n_heads * v_head_dim, 1))
    wo_out_size = size(wo, 1)
    decode_wo_buf = oneArray(zeros(Float16, wo_out_size, 1))

    # CPU scratchpads
    q_pe_cpu = zeros(Float32, n_heads * qk_rope_head_dim)
    k_pe_cpu = zeros(Float32, n_heads * qk_rope_head_dim)
    scores_cpu = zeros(Float32, max_seq)
    pb_cpu = zeros(Float32, max_seq)

    return MLAttention(q_a_proj, q_a_norm, q_b_proj, kv_a_proj_with_mqa, kv_a_norm, kv_b_proj, wo,
        n_heads, head_dim, q_lora_rank, kv_lora_rank, qk_rope_head_dim, v_head_dim, qk_nope_head_dim,
        softmax_scale, decode_q_latent, decode_q_latent_norm, decode_q_all, decode_kv_a,
        decode_kv_latent, decode_kv_latent_norm, decode_kv_b, decode_scores, decode_pb,
        decode_out, decode_wo_buf, q_pe_cpu, k_pe_cpu, scores_cpu, pb_cpu)
end

function (m::MLAttention)(x::oneMatrix{Float16}, pos::Int, rope::RotaryEmbedding, cache::KVCache)
    # x: (hidden_size, seq)
    seq = size(x, 2)
    T = Float16

    # Compute dimensions
    qk_pe_size = m.n_heads * m.qk_rope_head_dim  # Total RoPE dimensions
    qk_nope_size = m.n_heads * m.qk_nope_head_dim  # Total non-RoPE dimensions
    v_size = m.n_heads * m.v_head_dim

    if seq == 1
        # === Decode path (single token) ===

        # 1. Q projection: x -> q_latent -> q_latent_norm -> q_all
        mat_mul!(m.decode_q_latent, m.q_a_proj, x)
        q_latent_norm = m.q_a_norm(reshape(m.decode_q_latent, :))
        copyto!(m.decode_q_latent_norm, reshape(q_latent_norm, :, 1))
        mat_mul!(m.decode_q_all, m.q_b_proj, m.decode_q_latent_norm)

        # Split q_all into q_nope and q_pe
        q_pe = view(m.decode_q_all, 1:qk_pe_size, :)
        q_nope = view(m.decode_q_all, qk_pe_size+1:qk_pe_size+qk_nope_size, :)

        # 2. KV projection
        mat_mul!(m.decode_kv_a, m.kv_a_proj_with_mqa, x)
        copyto!(m.decode_kv_latent, view(m.decode_kv_a, 1:m.kv_lora_rank, :))
        k_pe = view(m.decode_kv_a, m.kv_lora_rank+1:m.kv_lora_rank+m.qk_rope_head_dim, :)

        # Normalize kv_latent and project to k_nope and v
        kv_latent_norm = m.kv_a_norm(reshape(m.decode_kv_latent, :))
        copyto!(m.decode_kv_latent_norm, reshape(kv_latent_norm, :, 1))
        mat_mul!(m.decode_kv_b, m.kv_b_proj, m.decode_kv_latent_norm)

        # Split kv_b into k_nope and v
        k_nope = view(m.decode_kv_b, 1:qk_nope_size, :)
        v = view(m.decode_kv_b, qk_nope_size+1:qk_nope_size+v_size, :)

        # 3. Apply RoPE to q_pe and k_pe
        q_pe_3d = reshape(q_pe, m.qk_rope_head_dim, m.n_heads, 1)
        k_pe_3d = reshape(k_pe, m.qk_rope_head_dim, m.n_heads, 1)
        rope(q_pe_3d, pos)
        rope(k_pe_3d, pos)

        # 4. Update KV cache
        # For MLA, we store full K (k_nope + k_pe) and V in the cache
        # K cache shape: (qk_nope_head_dim + qk_rope_head_dim, n_heads, max_seq)
        # But standard cache has: (head_dim, n_kv, max_seq)
        # We'll use a simplified approach: store k_nope and k_pe separately
        
        # Store k_nope in cache.k (reshaped to fit)
        # Store k_pe in cache.v's extra space (we'll use a workaround)
        # For now, store combined K in cache.k and V in cache.v
        k_combined = vcat(reshape(k_nope, :, 1), reshape(k_pe_3d, :, 1))
        v_combined = reshape(v, :, 1)
        
        # Update cache - note: this assumes cache can hold n_heads channels
        @views cache.k[:, 1, pos + 1] .= k_combined
        @views cache.v[:, 1, pos + 1] .= v_combined
        cache.pos = pos + 1

        # 5. Compute attention scores
        total_len = cache.pos
        k_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
        
        # Compute attention for each head
        fill!(m.decode_scores, T(0.0))
        fill!(m.decode_pb, T(0.0))

        for h in 1:m.n_heads
            # Get cached K and V for this head
            # K is stored as (k_nope + k_pe) for all positions
            k_cached = view(cache.k, (h-1)*k_head_dim+1:h*k_head_dim, 1, 1:total_len)
            v_cached = view(cache.v, (h-1)*m.v_head_dim+1:h*m.v_head_dim, 1, 1:total_len)
            
            # Current Q for this head
            q_h_nope = view(q_nope, (h-1)*m.qk_nope_head_dim+1:h*m.qk_nope_head_dim, :)
            q_h_pe = view(q_pe_3d, :, h, :)
            q_h_combined = vcat(vec(q_h_nope), vec(q_h_pe))

            # Score computation with numerical stability in Float32
            for p in 1:total_len
                k_p = view(k_cached, :, p)
                score = Float32(0.0)
                for d in 1:k_head_dim
                    score += Float32(q_h_combined[d]) * Float32(k_p[d])
                end
                m.scores_cpu[p] = score * Float32(m.softmax_scale)
            end

            # Stable softmax in Float32
            mx = maximum(m.scores_cpu[1:total_len])
            s = Float32(0.0)
            for p in 1:total_len
                m.pb_cpu[p] = exp(m.scores_cpu[p] - mx)
                s += m.pb_cpu[p]
            end
            inv_s = Float32(1.0) / s
            for p in 1:total_len
                m.pb_cpu[p] *= inv_s
            end
            copyto!(view(m.decode_pb, 1:total_len, h), m.pb_cpu[1:total_len])

            # Weighted sum of V
            out_h = view(m.decode_out, (h-1)*m.v_head_dim+1:h*m.v_head_dim, :)
            mul!(out_h, v_cached, view(m.decode_pb, 1:total_len, h:h))
        end

        # 6. Output projection
        result = mat_mul(m.wo, m.decode_out)
        return result

    else
        # === Prefill path (multiple tokens) ===
        # Similar to decode but processes all tokens in parallel

        # 1. Q projection
        q_latent = mat_mul(m.q_a_proj, x)
        q_latent_norm_2d = m.q_a_norm(q_latent)
        q_all = mat_mul(m.q_b_proj, q_latent_norm_2d)

        # Split q_all into q_nope and q_pe
        q_pe = view(q_all, 1:qk_pe_size, :)
        q_nope = view(q_all, qk_pe_size+1:qk_pe_size+qk_nope_size, :)

        # 2. KV projection
        kv_a = mat_mul(m.kv_a_proj_with_mqa, x)
        kv_latent = view(kv_a, 1:m.kv_lora_rank, :)
        k_pe = view(kv_a, m.kv_lora_rank+1:m.kv_lora_rank+m.qk_rope_head_dim, :)

        # Normalize kv_latent and project to k_nope and v
        kv_latent_norm_2d = m.kv_a_norm(kv_latent)
        kv_b = mat_mul(m.kv_b_proj, kv_latent_norm_2d)

        # Split kv_b into k_nope and v
        k_nope = view(kv_b, 1:qk_nope_size, :)
        v = view(kv_b, qk_nope_size+1:qk_nope_size+v_size, :)

        # 3. Apply RoPE to q_pe and k_pe
        q_pe_3d = reshape(q_pe, m.qk_rope_head_dim, m.n_heads, seq)
        k_pe_3d = reshape(k_pe, m.qk_rope_head_dim, m.n_heads, seq)
        rope(q_pe_3d, pos + 1)  # pos is 0-indexed, RoPE expects 1-indexed start
        rope(k_pe_3d, pos + 1)

        # 4. Update KV cache
        k_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
        for s in 1:seq
            for h in 1:m.n_heads
                k_nope_h = view(k_nope, (h-1)*m.qk_nope_head_dim+1:h*m.qk_nope_head_dim, s)
                k_pe_h = view(k_pe_3d, :, h, s)
                k_combined = vcat(vec(k_nope_h), vec(k_pe_h))
                v_h = view(v, (h-1)*m.v_head_dim+1:h*m.v_head_dim, s)
                
                cache_pos = pos + s
                @views cache.k[:, 1, cache_pos] .= k_combined
                @views cache.v[:, 1, cache_pos] .= v_h
            end
        end
        cache.pos = pos + seq

        # 5. Compute attention scores
        q_nope_3d = reshape(q_nope, m.qk_nope_head_dim, m.n_heads, seq)

        # Allocate output buffer
        out_all = zeros(Float32, m.v_head_dim, m.n_heads, seq)

        for h in 1:m.n_heads
            for s in 1:seq
                q_h_nope = view(q_nope_3d, :, h, s)
                q_h_pe = view(q_pe_3d, :, h, s)
                q_h_combined = vcat(vec(q_h_nope), vec(q_h_pe))

                scores = zeros(Float32, pos + s)
                for p in 1:(pos + s)
                    k_p = view(cache.k, (h-1)*k_head_dim+1:h*k_head_dim, 1, p)
                    score = Float32(0.0)
                    for d in 1:k_head_dim
                        score += Float32(q_h_combined[d]) * Float32(k_p[d])
                    end
                    scores[p] = score * Float32(m.softmax_scale)
                end

                # Stable softmax
                mx = maximum(scores)
                sum_exp = Float32(0.0)
                for p in 1:(pos + s)
                    scores[p] = exp(scores[p] - mx)
                    sum_exp += scores[p]
                end
                scores ./= sum_exp

                # Weighted sum of V
                for d in 1:m.v_head_dim
                    val = Float32(0.0)
                    for p in 1:(pos + s)
                        v_p = cache.v[(h-1)*m.v_head_dim+d, 1, p]
                        val += Float32(v_p) * scores[p]
                    end
                    out_all[d, h, s] = val
                end
            end
        end

        # 6. Output projection
        out_tensor = oneArray(Float16.(reshape(out_all, m.v_head_dim * m.n_heads, seq)))
        result = mat_mul(m.wo, out_tensor)
        return result
    end
end

function reset_states!(m::MLAttention)
    # MLAttention has no persistent state to reset (KV cache is external)
    return nothing
end

# --- Gated Delta Net (SSM Layer) ---
# Reference: qwen35.cpp build_layer_attn_linear
# Uses Float32 for all internal computations for numerical stability

struct GatedDeltaNet
 index::Int # Layer index

 # Weights
 in_proj::Union{oneMatrix{Float16},QuantMatrix} # wqkv: (hidden, 6144)
 gate_proj::Union{oneMatrix{Float16},QuantMatrix} # wqkv_gate: (hidden, d_inner=2048)
 ssm_out::Union{oneMatrix{Float16},QuantMatrix} # (d_inner, hidden)
 ssm_conv1d_weight::oneMatrix{Float16} # (conv_kernel, conv_channels)
 ssm_conv1d_weight_cpu::Matrix{Float32} # CPU copy for fast access
 ssm_alpha_weight::oneMatrix{Float16} # (hidden, num_v_heads)
 ssm_beta_weight::oneMatrix{Float16} # (hidden, num_v_heads)
 ssm_alpha_weight_cpu::Matrix{Float32} # CPU copy
 ssm_beta_weight_cpu::Matrix{Float32} # CPU copy
 ssm_a::oneVector{Float16} # (num_v_heads,)
 ssm_a_cpu::Vector{Float32} # CPU copy
 ssm_dt_bias::oneVector{Float16} # (num_v_heads,)
 ssm_dt_bias_cpu::Vector{Float32} # CPU copy
 ssm_norm::RMSNorm

 # Dimensions
 num_v_heads::Int # = 16 (ssm_time_step_rank)
 num_k_heads::Int # = 16 (ssm_group_count)
 head_k_dim::Int # = 128 (ssm_state_size)
 head_v_dim::Int # = 128 (d_inner / num_v_heads)
 d_inner::Int # = 2048
 conv_channels::Int # = 3 * d_inner = 6144
 conv_kernel::Int # = 4

 # CPU state buffers (Float32 for numerical stability)
 conv_state::Matrix{Float32} # (conv_channels, conv_kernel) - ring buffer
 h::Array{Float32,3} # (head_v_dim, head_k_dim, num_v_heads) - SSM state

 # Scratchpad buffers (2D column matrices for GPU matmul compatibility)
 qkv_proj::oneMatrix{Float16} # (conv_channels, 1)
 z_buf::oneMatrix{Float16} # (d_inner, 1)
 x_conv::oneMatrix{Float16} # (conv_channels, 1)
 y_all::oneMatrix{Float16} # (d_inner, 1)
 branch_out::oneMatrix{Float16} # (d_inner, 1)

 # CPU scratchpads
 x_norm_cpu::Vector{Float16}
 x_norm_cpu32::Vector{Float32}
 x_conv_cpu::Vector{Float32}
 y_all_cpu::Vector{Float32}
 z_buf_cpu::Vector{Float32} # (d_inner,) for gate computation
 alpha_proj::Vector{Float32}
 beta_proj::Vector{Float32}
 q_norm_buf::Vector{Float32}
 k_norm_buf::Vector{Float32}
 tmp_head::Vector{Float32}
 sk_buf::Vector{Float32}
end

function GatedDeltaNet(index::Int, in_proj, gate_proj, ssm_out, ssm_conv1d, ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias, ssm_norm,
    config::QwenConfig)
 num_v_heads = config.ssm_time_step_rank
 num_k_heads = config.ssm_group_count
 head_k_dim = config.ssm_state_size
 head_v_dim = config.ssm_inner_size ÷ num_v_heads
 d_inner = config.ssm_inner_size
 # conv_channels = d_inner + 2 * num_k_heads * head_k_dim (from llama.cpp qwen35.cpp:253)
 conv_channels = d_inner + 2 * num_k_heads * head_k_dim
 conv_kernel = config.ssm_conv_kernel

    # Convert weights to proper types
    ssm_conv1d_gpu = ssm_conv1d isa oneArray ? ssm_conv1d : oneArray(Float16.(ssm_conv1d))
    ssm_conv1d_cpu = Array(Float32.(ssm_conv1d_gpu))

    ssm_alpha_cpu = Array(Float32.(ssm_alpha))
    ssm_beta_cpu = Array(Float32.(ssm_beta))
    ssm_a_cpu = Array(Float32.(ssm_a))
    ssm_dt_bias_cpu = Array(Float32.(ssm_dt_bias))

    # State buffers
 conv_state = zeros(Float32, conv_channels, conv_kernel)
 h = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)

    # GPU scratchpads (2D column matrices for GPU matmul compatibility)
    qkv_proj = oneArray(zeros(Float16, conv_channels, 1))
    z_buf = oneArray(zeros(Float16, d_inner, 1))
    x_conv = oneArray(zeros(Float16, conv_channels, 1))
    y_all = oneArray(zeros(Float16, d_inner, 1))
    branch_out = oneArray(zeros(Float16, d_inner, 1))

 # CPU scratchpads
 x_norm_cpu = zeros(Float16, config.hidden_size)
 x_norm_cpu32 = zeros(Float32, config.hidden_size)
 x_conv_cpu = zeros(Float32, conv_channels)
 y_all_cpu = zeros(Float32, d_inner)
 z_buf_cpu = zeros(Float32, d_inner)
 alpha_proj = zeros(Float32, num_v_heads)
 beta_proj = zeros(Float32, num_v_heads)
 q_norm_buf = zeros(Float32, head_k_dim)
 k_norm_buf = zeros(Float32, head_k_dim)
 tmp_head = zeros(Float32, head_v_dim)
 sk_buf = zeros(Float32, head_v_dim)

 return GatedDeltaNet(index, in_proj, gate_proj, ssm_out, ssm_conv1d_gpu, ssm_conv1d_cpu,
 ssm_alpha, ssm_beta, ssm_alpha_cpu, ssm_beta_cpu,
 ssm_a, ssm_a_cpu, ssm_dt_bias, ssm_dt_bias_cpu, ssm_norm,
 num_v_heads, num_k_heads, head_k_dim, head_v_dim, d_inner, conv_channels, conv_kernel,
 conv_state, h, qkv_proj, z_buf, x_conv, y_all, branch_out,
 x_norm_cpu, x_norm_cpu32, x_conv_cpu, y_all_cpu, z_buf_cpu,
 alpha_proj, beta_proj, q_norm_buf, k_norm_buf, tmp_head, sk_buf)
end

function reset_states!(m::GatedDeltaNet)
    fill!(m.conv_state, Float32(0.0))
    fill!(m.h, Float32(0.0))
end

function (m::GatedDeltaNet)(x::AbstractArray{Float16,2}, pos::Int, rope::RotaryEmbedding, cache)
 hidden = size(x, 1)
 T = Float16

 # 1. Input projections (weights are already transposed by get_weight)
 # Explicitly zero output buffers before use for numerical safety
 fill!(m.qkv_proj, Float16(0.0))
 fill!(m.z_buf, Float16(0.0))
 mat_mul!(m.qkv_proj, m.in_proj, x)
 mat_mul!(m.z_buf, m.gate_proj, x)

 # 2. Update conv state (ring buffer)
 qkv_cpu = vec(Float32.(Array(m.qkv_proj))) # Copy GPU result to CPU as 1D vector
 if m.conv_kernel > 1
 m.conv_state[:, 1:(m.conv_kernel-1)] .= m.conv_state[:, 2:m.conv_kernel]
 end
 m.conv_state[:, m.conv_kernel] .= qkv_cpu
 
 # 3. Compute convolution (store result in x_conv_cpu)
 fill!(m.x_conv_cpu, Float32(0.0))
 for k in 1:m.conv_kernel
 @inbounds for c in 1:m.conv_channels
 m.x_conv_cpu[c] += m.conv_state[c, k] * m.ssm_conv1d_weight_cpu[k, c]
 end
 end

    # 4. SiLU on convolved input
    @inbounds for c in 1:m.conv_channels
        v = m.x_conv_cpu[c]
        m.x_conv_cpu[c] = v * (Float32(1.0) / (Float32(1.0) + exp(-v)))
    end

 # 5. Split into Q, K, V
 qk_size = m.head_k_dim * m.num_k_heads
 q_all = reshape(view(m.x_conv_cpu, 1:qk_size), m.head_k_dim, m.num_k_heads)
 k_all = reshape(view(m.x_conv_cpu, qk_size+1:2*qk_size), m.head_k_dim, m.num_k_heads)
 # V has different number of heads than Q/K!
 v_all = reshape(view(m.x_conv_cpu, 2*qk_size+1:2*qk_size+m.d_inner), m.head_v_dim, m.num_v_heads)

    # 6. Alpha/beta projections
    copyto!(m.x_norm_cpu, x)
    @inbounds for i in eachindex(m.x_norm_cpu)
        m.x_norm_cpu32[i] = Float32(m.x_norm_cpu[i])
    end

    mul!(m.alpha_proj, m.ssm_alpha_weight_cpu, m.x_norm_cpu32)
    mul!(m.beta_proj, m.ssm_beta_weight_cpu, m.x_norm_cpu32)

 # 7. Process each head (autoregressive delta net)
 fill!(m.y_all_cpu, Float32(0.0))
 scale = Float32(1.0 / sqrt(Float64(m.head_k_dim)))
 
 # When num_v_heads != num_k_heads, we need to broadcast Q/K
 # num_v_heads is typically larger than num_k_heads
 @inbounds for h in 1:m.num_v_heads
 # Map v_head to k_head (each k_head serves multiple v_heads)
 g = ((h - 1) % m.num_k_heads) + 1
 
 qg = view(q_all, :, g) # (head_k_dim,)
 kg = view(k_all, :, g) # (head_k_dim,)
 vg = view(v_all, :, h) # (head_v_dim,)
 
 # Q/K L2 normalization (llama.cpp uses L2 norm)
 q_norm_sq = mapreduce(v -> Float32(v)^2, +, qg)
 k_norm_sq = mapreduce(v -> Float32(v)^2, +, kg)
 q_norm_val = sqrt(q_norm_sq + Float32(1e-6))
 k_norm_val = sqrt(k_norm_sq + Float32(1e-6))
 
 for j in 1:m.head_k_dim
 m.q_norm_buf[j] = Float32(qg[j]) / q_norm_val * scale
 m.k_norm_buf[j] = Float32(kg[j]) / k_norm_val
 end
 
 # Compute gate values (decay g and input beta)
 # gate = exp(softplus(alpha + dt_bias) * a)
 # beta = sigmoid(beta)
 alpha_val = Float64(m.alpha_proj[h]) + Float64(m.ssm_dt_bias_cpu[h])
 # Numerical stability: clamp alpha_val to prevent overflow
 alpha_val = clamp(alpha_val, -20.0, 20.0)
 softplus_alpha = log(Float64(1.0) + exp(alpha_val)) # softplus
 # Clamp softplus_alpha to prevent decay explosion/underflow
 softplus_alpha = clamp(softplus_alpha, -10.0, 10.0)
 decay = Float32(exp(softplus_alpha * Float64(m.ssm_a_cpu[h])))
 # Clamp decay to [0, 1] since it's a decay factor
 decay = clamp(decay, Float32(0.0), Float32(1.0))
 beta_val = Float64(m.beta_proj[h])
 # Clamp beta_val for numerical stability
 beta_val = clamp(beta_val, -20.0, 20.0)
 beta = Float32(1.0 / (Float64(1.0) + exp(-beta_val))) # sigmoid
 
 # Get state for this head: (head_v_dim, head_v_dim)
 state = view(m.h, :, :, h)
 
 # State decay: s = s * decay
 state .*= decay
 
 # sk = state * k_norm_buf (matrix-vector multiplication)
 # state is (S_v, S_v), k_norm_buf is (S_v,), result is (S_v,)
 mul!(m.sk_buf, state, m.k_norm_buf)
 
 # d = (v - sk) * beta
 for i in 1:m.head_v_dim
 m.tmp_head[i] = beta * (Float32(vg[i]) - m.sk_buf[i])
 end
 
 # state += tmp_head * k_norm_buf' (outer product)
 BLAS.ger!(Float32(1.0), m.tmp_head, m.k_norm_buf, state)
 
 # Output: o = state * q_norm_buf
 yg = view(m.y_all_cpu, (h-1)*m.head_v_dim+1:h*m.head_v_dim)
 mul!(yg, state, m.q_norm_buf)
 end

 # 8. Reshape and apply SSM norm
 y_reshaped = reshape(m.y_all_cpu, m.head_v_dim, m.num_k_heads)
 ssm_norm_w_cpu = Float32.(Array(m.ssm_norm.weight))
 rmsnorm_cpu!(y_reshaped, y_reshaped, ssm_norm_w_cpu, Float32(m.ssm_norm.eps))

 # 9. SiLU gate on z
 copyto!(m.z_buf_cpu, vec(m.z_buf))
 @inbounds for i in 1:m.d_inner
 v = m.z_buf_cpu[i]
 m.y_all_cpu[i] = m.y_all_cpu[i] * (v * (Float32(1.0) / (Float32(1.0) + exp(-v))))
 end

 # 10. Output projection (ssm_out is (hidden, d_inner) after get_weight transpose)
 fill!(m.branch_out, Float16(0.0))
 copyto!(m.branch_out, reshape(Float16.(m.y_all_cpu), :, 1))
 oneAPI.synchronize()  # Ensure copy is complete before matmul
 result = mat_mul(m.ssm_out, m.branch_out)
 oneAPI.synchronize()  # Ensure result is computed before returning

 return result
end

# --- Decoder Layer ---
struct DecoderLayer
    in_norm::RMSNorm
    op::Union{GatedDeltaNet,FullAttention,MLAttention}
    post_norm::RMSNorm
    mlp::Union{MLP,MoE}
    is_ssm::Bool
end

function (layer::DecoderLayer)(x::AbstractArray{Float16,2}, pos::Int, rope::RotaryEmbedding, cache)
    h = layer.in_norm(x)
    h = layer.op(h, pos, rope, cache)
    x .+= h
    h = layer.post_norm(x)
    h = layer.mlp(h, cache)
    x .+= h
    return x
end

# --- Model ---
struct QwenModel
    config::QwenConfig
    embed::Matrix{Float16} # CPU-based (kept for compatibility)
    layers::Vector{DecoderLayer}
    final_norm::RMSNorm
    lm_head::Matrix{Float16} # CPU-based (kept for compatibility)
    rope::RotaryEmbedding
    mmproj::Union{Dict{String,Any},Nothing}
end

# Free all GPU memory associated with a QwenModel (weights, norms, etc.)
# Note: This frees the GPU arrays but keeps the model structure for reloading
function free_model_gpu!(model::QwenModel)
    # Free embedding
    try
        if isdefined(model, :embed) && model.embed !== nothing
            # embed is CPU-based, no GPU memory to free
        end
    catch e
        @warn "Error freeing embed" exception = (e, catch_backtrace())
    end

    # Free layers - each layer has attention/SSM and MLP with GPU arrays
    for layer in model.layers
        try
            # Free in_norm and post_norm
            if isdefined(layer, :in_norm) && isdefined(layer.in_norm, :weight)

            end
            if isdefined(layer, :post_norm) && isdefined(layer.post_norm, :weight)

            end
        catch e
            @warn "Error freeing layer norms" exception = (e, catch_backtrace())
        end

        try
            # Free attention/SSM op
            if layer.op isa FullAttention
                # Free pre-allocated GPU buffers in FullAttention
                op = layer.op
                for field in fieldnames(FullAttention)
                    if field in [:decode_q_full, :decode_k, :decode_v, :decode_combined,
                        :decode_scores, :decode_pb, :decode_out_h, :decode_wo_buf,
                        :prefill_scores, :prefill_pb]
                        try
                            buf = getfield(op, field)
                            if buf !== nothing

                            end
                        catch
                        end
                    end
                end
            elseif layer.op isa GatedDeltaNet
                # Free GPU buffers in GatedDeltaNet
                op = layer.op
                for field in fieldnames(GatedDeltaNet)
                    if field in [:decode_beta, :decode_alpha, :decode_decay_gate,
                        :decode_conv_out, :decode_z, :decode_output_normed,
                        :decode_gated, :prefill_beta, :prefill_alpha]
                        try
                            buf = getfield(op, field)
                            if buf !== nothing

                            end
                        catch
                        end
                    end
                end
            end
        catch e
            @warn "Error freeing layer op" exception = (e, catch_backtrace())
        end

        try
            # Free MLP
            mlp = layer.mlp
            for field in fieldnames(MLP)
                try
                    w = getfield(mlp, field)
                    if w !== nothing

                    end
                catch
                end
            end
        catch e
            @warn "Error freeing MLP" exception = (e, catch_backtrace())
        end
    end

    # Free final_norm
    try
        if isdefined(model, :final_norm) && isdefined(model.final_norm, :weight)

        end
    catch e
        @warn "Error freeing final_norm" exception = (e, catch_backtrace())
    end

    # lm_head is CPU-based, no GPU memory to free

    # Free global lookup tables if they were initialized
    # This prevents memory leaks across multiple model loads
    try
        if isassigned(IQ2XXS_GRID_GPU)

            # We don't null out Ref, as it's a const, but the underlying device memory is freed
        end
        if isassigned(KSIGNS_IQ2XS_GPU)

        end
        if isassigned(KMASK_IQ2XS_GPU)

        end
    catch
        # Best effort cleanup
    end

    return nothing
end

function forward!(model::QwenModel, tokens::Vector{Int}, pos::Int, caches::Vector{<:KVCache}; gc_interval::Int=0)
 try
 seq_len = length(tokens)

 # For prefill (seq_len > 1), process tokens one at a time
 # This is needed because SSM layers have stateful buffers sized for single tokens
 if seq_len == 1
 # Single token path - process directly
 indices = tokens
 emb_rows = model.embed[:, indices]
 x = oneArray(Float16.(emb_rows))

 for (i, layer) in enumerate(model.layers)
 x = layer(x, pos, model.rope, caches[i])
 # Optional GC during inference (disabled by default to avoid stuttering)
 if gc_interval > 0 && i % gc_interval == 0
 GC.gc(false)
 end
 end

 x_normed = model.final_norm(x)
 oneAPI.synchronize() # Ensure GPU computation is complete
 x_final = collect(x_normed)
 logits = (model.lm_head') * x_final

 # Sanitize logits - replace NaN/Inf with 0
 @inbounds for i in eachindex(logits)
 if !isfinite(logits[i])
 logits[i] = 0.0
 end
 end

 return Float16.(logits)
 else
 # Prefill path - process tokens one at a time
 all_logits = nothing
 for t in 1:seq_len
 tok = tokens[t]
 curr_pos = pos + t - 1
 
 emb_row = model.embed[:, tok]
 x = oneArray(reshape(Float16.(emb_row), :, 1)) # Make it a 2D column vector
 
 for (i, layer) in enumerate(model.layers)
 x = layer(x, curr_pos, model.rope, caches[i])
 end
 
 x_normed = model.final_norm(x)
 oneAPI.synchronize() # Ensure GPU computation is complete
 x_final = collect(x_normed)
 logits_t = (model.lm_head') * x_final
 
 # Sanitize logits - replace NaN/Inf with 0
 @inbounds for i in eachindex(logits_t)
 if !isfinite(logits_t[i])
 logits_t[i] = 0.0
 end
 end
 
 if all_logits === nothing
 all_logits = Float16.(zeros(size(logits_t, 1), seq_len))
 end
 all_logits[:, t] = Float16.(logits_t)
 end
 return all_logits
 end
 catch e
 @error "forward! failed" exception = (e, catch_backtrace())
        # Cleanup on error - free caches and force GC
        try
            free_all_kv_caches!(caches)
        catch
        end
        try
            GC.gc(true)
        catch
        end
        try
            nothing
        catch
        end
        rethrow(e)
    end
end


function reset_states!(model::QwenModel)
    for layer in model.layers
        reset_states!(layer.op)
        reset_states!(layer.mlp)
    end
end

# --- Model Display ---

function Base.show(io::IO, ::MIME"text/plain", model::QwenModel)
    config = model.config
    arch = config.architecture

    # Header
    println(io, "QwenModel (GPU backend)")
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

    # SSM params (only relevant for Qwen3.5-style SSM models)
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

    # Device info
    try
        device_str = string(oneAPI.device())
        println(io, "└─ Device: ", device_str)
    catch
        println(io, "└─ Device: unknown (oneAPI error)")
    end
end

function Base.show(io::IO, model::QwenModel)
    show(io, MIME"text/plain"(), model)
end

end # module
