"""
GPUCommon — Unified KernelAbstractions.jl kernels for Inferno.jl

Provides fused GPU kernels for all model backends (Gemma4, Qwen3.5).
All kernels use KernelAbstractions.jl for portability and performance.

TYPE-GENERIC: All kernels now work with Float16, Float32, or any AbstractFloat.
For Intel Arc, using Float16 weights and activations gives ~2x speedup
over Float32 because Arc has native FP16 throughput.
"""
module GPUCommon

using KernelAbstractions: @kernel, @index, @localmem, @synchronize
using LinearAlgebra
using oneAPI
using oneAPI: oneAPIBackend

# Module-level backend for KA kernel calls
const _GPU_BACKEND = oneAPIBackend()

export rmsnorm_kernel!, gelu_kernel!, residual_add_kernel!, apply_rope_kernel!
export softmax_kernel!, matmul_vec_kernel!, silu_kernel!, sigmoid_kernel!
export batched_attention_scores_kernel!, batched_softmax_kernel!, batched_ssm_state_kernel!
export rmsnorm_gpu!, gelu_gpu!, residual_add_gpu!, apply_rope_gpu!, softmax_gpu!, silu_gpu!, sigmoid_gpu!
export batched_attention_scores!, batched_softmax!, batched_ssm_state_update!, batched_ssm_output_sum!, fused_attention_forward!
export write_kv_cache_kernel!, write_kv_cache_gpu!

# ============================================================
# RMSNorm Kernel (type-generic)
# ============================================================
"""
 rmsnorm_kernel!(out, x, weight, eps, sum_sq)

In-place RMSNorm: out[i] = x[i] * inv_rms * weight[i]
where inv_rms = 1 / sqrt(sum_sq / n + eps)

The sum_sq must be pre-computed (reduction step) to avoid O(n²) work.
Type-generic — works with Float16, Float32, etc.
"""
@kernel function rmsnorm_kernel!(out, x, weight, eps, sum_sq)
    i = @index(Global, Linear)
    n = length(x)
    T = eltype(x)
    inv_rms = one(T) / sqrt(sum_sq / n + eps)
    @inbounds out[i] = x[i] * inv_rms * weight[i]
end

function rmsnorm_gpu!(out::AbstractArray, x::AbstractArray, weight::AbstractArray, eps::AbstractFloat)
    n = length(x)
    if n == 0 return out end
    # GPU-safe sum of squares (broadcast, not BLAS)
    # oneAPI supports broadcast .^ and sum natively without oneMKL
    x_sq = x .^ 2
    sum_sq = oneAPI.oneAPI.sum(x_sq)
    # Normalize using the pre-computed sum
    kfn = rmsnorm_kernel!(_GPU_BACKEND)
    kfn(out, x, weight, eps, sum_sq; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# GELU Kernel (with tanh approximation — type generic)
# ============================================================
@inline function gelu_tanh(x)
    T = typeof(x)
    cdf = 0.5 * (one(T) + tanh(T(0.7978845608028654) * (x + T(0.044715) * x^3)))
    return x * cdf
end

@kernel function gelu_kernel!(out, x)
    i = @index(Global, Linear)
    @inbounds out[i] = gelu_tanh(x[i])
end

function gelu_gpu!(out::AbstractArray, x::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kfn = gelu_kernel!(_GPU_BACKEND)
    kfn(out, x; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Residual Add Kernel (type generic)
# ============================================================
@kernel function residual_add_kernel!(out, x, residual)
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] + residual[i]
end

function residual_add_gpu!(out::AbstractArray, x::AbstractArray, residual::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kfn = residual_add_kernel!(_GPU_BACKEND)
    kfn(out, x, residual; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Rotary Embedding Kernel (type generic)
# ============================================================
@kernel function apply_rope_kernel!(x, cos, sin, rotary_dim)
    i = @index(Global, Linear)
    # i corresponds to (head, dim) flattened
end

function apply_rope_gpu!(out::AbstractArray, x::AbstractArray, cos_cache::AbstractArray, sin_cache::AbstractArray, 
                        pos::Int, head_dim::Int, rotary_dim::Int)
    # Placeholder for type-generic RoPE
    # Production implementation would use the same logic as in Qwen35GPU.jl
    return out
end

# ============================================================
# Softmax Kernel (type generic)
# ============================================================
@kernel function softmax_kernel!(out, x)
    i = @index(Global, Linear)
    # Single-threaded stable softmax
    if i == 1
        n = length(x)
        T = eltype(x)
        max_val = x[1]
        for j in 2:n
            if x[j] > max_val
                max_val = x[j]
            end
        end
        sum_exp = zero(T)
        for j in 1:n
            out[j] = exp(x[j] - max_val)
            sum_exp += out[j]
        end
        inv_sum = one(T) / sum_exp
        for j in 1:n
            out[j] *= inv_sum
        end
    end
end

function softmax_gpu!(out::AbstractArray, x::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kfn = softmax_kernel!(_GPU_BACKEND)
    kfn(out, x; ndrange=(1,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# SiLU Kernel (type generic)
# ============================================================
@kernel function silu_kernel!(out, x)
    i = @index(Global, Linear)
    T = eltype(x)
    @inbounds out[i] = x[i] / (one(T) + exp(-x[i]))
end

function silu_gpu!(out::AbstractArray, x::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kfn = silu_kernel!(_GPU_BACKEND)
    kfn(out, x; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Sigmoid Kernel (type generic)
# ============================================================
@kernel function sigmoid_kernel!(out, x)
    i = @index(Global, Linear)
    T = eltype(x)
    @inbounds out[i] = one(T) / (one(T) + exp(-x[i]))
end

function sigmoid_gpu!(out::AbstractArray, x::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kfn = sigmoid_kernel!(_GPU_BACKEND)
    kfn(out, x; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Matmul Vector Kernel (type generic)
# ============================================================
@kernel function matmul_vec_kernel!(y, A, x)
    i = @index(Global, Linear)
    T = eltype(y)
    acc = zero(T)
    for j in 1:size(A, 2)
        acc += A[i, j] * x[j]
    end
    @inbounds y[i] = acc
end

# ============================================================
# Batched Attention Scores Kernel (type generic)
# Computes dot(Q[h], K_cache[h, :, s]) for all heads in parallel
# ============================================================
@kernel function batched_attention_scores_kernel!(scores, q, k_cache, n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * seq_len
    T = eltype(scores)
    if idx <= n_total
        h = (idx - 1) ÷ seq_len  # 0-based head index
        s = (idx - 1) % seq_len  # 0-based position
        kv_h = h ÷ n_groups
        q_off = h * head_dim
        k_off = kv_h * head_dim
        k_pos_off = s
        
        dot_sum = zero(T)
        for d in 0:(head_dim-1)
            q_val = q[q_off + d + 1]
            k_val = k_cache[k_off + d + 1, k_pos_off + 1]
            dot_sum += q_val * k_val
        end
        
        scores[idx] = dot_sum / sqrt(T(head_dim))
    end
end

function batched_attention_scores!(scores::AbstractArray{T,2}, q::AbstractArray{T,1}, 
                                   k_cache::AbstractArray{T,2}, n_heads::Int, head_dim::Int, 
                                   seq_len::Int, n_groups::Int) where T
    n_total = n_heads * seq_len
    kfn = batched_attention_scores_kernel!(_GPU_BACKEND)
    kfn(scores, q, k_cache, n_heads, head_dim, seq_len, n_groups; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return scores
end

# ============================================================
# Batched Softmax Kernel (type generic)
# ============================================================
@kernel function batched_softmax_kernel!(scores, n_heads::Int, seq_len::Int)
    idx = @index(Global, Linear)
    T = eltype(scores)
    if idx <= n_heads * seq_len
        h = (idx - 1) ÷ seq_len
        s = (idx - 1) % seq_len
        row_start = h * seq_len + 1
        
        # Find max
        max_val = typemin(T)
        for j in 0:(seq_len-1)
            val = scores[j * n_heads + h + 1]
            if val > max_val
                max_val = val
            end
        end
        
        # Compute exp and sum
        exp_val = exp(scores[idx] - max_val)
        
        sum_exp = zero(T)
        for j in 0:(seq_len-1)
            sum_exp += exp(scores[j * n_heads + h + 1] - max_val)
        end
        
        # Normalize
        scores[idx] = exp_val / sum_exp
    end
end

function batched_softmax!(scores::AbstractArray{T,2}, n_heads::Int, seq_len::Int) where T
    n_total = n_heads * seq_len
    kfn = batched_softmax_kernel!(_GPU_BACKEND)
    kfn(scores, n_heads, seq_len; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return scores
end

# ============================================================
# Batched SSM State Update Kernel (type generic)
# ============================================================
@kernel function batched_ssm_state_kernel!(h_state, decay, beta, x_conv, n_v::Int, head_v::Int, head_k::Int)
    idx = @index(Global, Linear)
    n_total = n_v * head_v * head_k
    T = eltype(h_state)
    if idx <= n_total
        v_head = (idx - 1) ÷ (head_v * head_k)
        rem = (idx - 1) % (head_v * head_k)
        h_dim = rem ÷ head_k
        k_dim = rem % head_k
        
        vb = v_head * head_v
        
        decay_val = decay[v_head + 1]
        beta_val = beta[vb + h_dim + 1]
        x_val = x_conv[vb + h_dim + 1]
        
        old_state = h_state[h_dim + 1, k_dim + 1, v_head + 1]
        new_state = decay_val * old_state + beta_val * x_val
        
        h_state[h_dim + 1, k_dim + 1, v_head + 1] = new_state
    end
end

function batched_ssm_state_update!(h_state::AbstractArray{T,3}, decay::AbstractArray{T,1},
                                   beta::AbstractArray{T,1}, x_conv::AbstractArray{T,1},
                                   n_v::Int, head_v::Int, head_k::Int) where T
    n_total = n_v * head_v * head_k
    kfn = batched_ssm_state_kernel!(_GPU_BACKEND)
    kfn(h_state, decay, beta, x_conv, n_v, head_v, head_k; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return h_state
end

# ============================================================
# Batched SSM Output Sum Kernel (type generic)
# ============================================================
@kernel function batched_ssm_output_kernel!(y_out, h_state, n_v::Int, head_v::Int, head_k::Int)
    idx = @index(Global, Linear)
    T = eltype(y_out)
    if idx <= n_v * head_v
        v_head = (idx - 1) ÷ head_v
        h_dim = (idx - 1) % head_v
        
        sum_val = zero(T)
        for k in 0:(head_k-1)
            sum_val += h_state[h_dim + 1, k + 1, v_head + 1]
        end
        
        y_out[idx] = sum_val
    end
end

function batched_ssm_output_sum!(y_out::AbstractArray{T,1}, h_state::AbstractArray{T,3},
                                 n_v::Int, head_v::Int, head_k::Int) where T
    n_total = n_v * head_v
    kfn = batched_ssm_output_kernel!(_GPU_BACKEND)
    kfn(y_out, h_state, n_v, head_v, head_k; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return y_out
end

# ============================================================
# KV Cache Write Kernel (type generic)
# Single kernel dispatch for all heads
# ============================================================
@kernel function write_kv_cache_kernel!(k_cache, v_cache, k, v, head_dim::Int, pos::Int)
    idx = @index(Global, Linear)
    n = length(k)
    if idx <= n
        h = (idx - 1) ÷ head_dim
        d = (idx - 1) % head_dim
        @inbounds k_cache[idx, pos+1] = k[idx]
        @inbounds v_cache[idx, pos+1] = v[idx]
    end
end

function write_kv_cache_gpu!(k_cache, v_cache, k, v, head_dim::Int, pos::Int)
    n = length(k)
    n == 0 && return nothing
    kfn = write_kv_cache_kernel!(_GPU_BACKEND)
    kfn(k_cache, v_cache, k, v, head_dim, pos; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return nothing
end

# ============================================================
# Fused Attention Kernel (scores + softmax + weighted sum)
# Combines the three attention steps into a single kernel launch.
# Each thread handles one (head, dim) pair.
# Recomputes scores per position to avoid intermediate buffer.
# Type generic — works with Float16, Float32, etc.
# ============================================================
@kernel function fused_attention_forward_kernel!(attn_out, q, k_cache, v_cache, n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * head_dim
    T = eltype(attn_out)
    if idx <= n_total
        h = (idx - 1) ÷ head_dim
        d = (idx - 1) % head_dim
        kv_h = h ÷ n_groups
        q_off = h * head_dim
        k_off = kv_h * head_dim
        v_off = kv_h * head_dim
        
        # Phase 1: Compute all scores for this head and track max
        max_val = typemin(T)
        for s in 0:(seq_len-1)
            dot_sum = zero(T)
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(T(head_dim))
            if score > max_val
                max_val = score
            end
        end
        
        # Phase 2: Compute sum of exp(score - max_val)
        sum_exp = zero(T)
        for s in 0:(seq_len-1)
            # Recompute score for this position
            dot_sum = zero(T)
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(T(head_dim))
            sum_exp += exp(score - max_val)
        end
        
        # Phase 3: Compute weighted sum for this dimension
        acc = zero(T)
        for s in 0:(seq_len-1)
            # Recompute score for this position
            dot_sum = zero(T)
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(T(head_dim))
            prob = exp(score - max_val) / sum_exp
            v_val = v_cache[v_off + d + 1, s + 1]
            acc += prob * v_val
        end
        
        attn_out[idx] = acc
    end
end

function fused_attention_forward!(attn_out::AbstractArray{T,1}, q::AbstractArray{T,1}, 
                                  k_cache::AbstractArray{T,2}, v_cache::AbstractArray{T,2},
                                  n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int) where T
    n_total = n_heads * head_dim
    if n_total == 0
        return attn_out
    end
    kfn = fused_attention_forward_kernel!(_GPU_BACKEND)
    kfn(attn_out, q, k_cache, v_cache, n_heads, head_dim, seq_len, n_groups; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return attn_out
end

end # module GPUCommon
