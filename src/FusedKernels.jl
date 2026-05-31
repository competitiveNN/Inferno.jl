module FusedKernels

using KernelAbstractions: @kernel, @index, @synchronize
using oneAPI
using oneAPI: oneAPIBackend
using LinearAlgebra

# Module-level backend for KA kernel calls
const _GPU_BACKEND = oneAPIBackend()

export fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!
export fused_ssm_gate_sigmoid!, fused_ssm_decay!, gpu_argmax!, gpu_sample!

# ============================================================
# L2 Norm (in-place) — type-generic version.
# Uses GPU dot for reduction then a 1-line scaling kernel.
# ============================================================
@kernel function l2norm_scale_kernel!(x, inv_norm)
    i = @index(Global, Linear)
    @inbounds x[i] = x[i] * inv_norm
end

function fused_l2norm!(x::AbstractArray{T,1}, eps::AbstractFloat) where T <: AbstractFloat
    n = length(x)
    n == 0 && return x
    ss = dot(x, x)
    inv_norm = one(T) / sqrt(ss + eps)
    kfn = l2norm_scale_kernel!(_GPU_BACKEND)
    kfn(x, inv_norm; ndrange=(n,))
    return x
end

# ============================================================
# Attention weighted sum across all heads (type generic)
# Replaces per-head loop that allocates weighted_v temporary.
# ============================================================
@kernel function attention_weighted_sum_kernel!(attn_out, scores, v_cache, n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * head_dim
    if idx <= n_total
        h = (idx - 1) ÷ head_dim
        d = (idx - 1) % head_dim
        kv_h = h ÷ n_groups
        k_off = kv_h * head_dim + d
        acc = zero(eltype(attn_out))
        for s in 1:seq_len
            acc += scores[h + 1, s] * v_cache[k_off + 1, s]
        end
        attn_out[idx] = acc
    end
end

function fused_attention_weighted_sum!(
    attn_out::AbstractArray{T,1},
    scores::AbstractArray{T,2},
    v_cache::AbstractArray{T,2},
    n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int
) where T
    n_total = n_heads * head_dim
    n_total == 0 && return attn_out
    kfn = attention_weighted_sum_kernel!(_GPU_BACKEND)
    kfn(attn_out, scores, v_cache, n_heads, head_dim, seq_len, n_groups; ndrange=(n_total,))
    return attn_out
end

# ============================================================
# MLP gate multiply: gate = SiLU(gate) .* up (in-place on gate)
# ============================================================
@kernel function mlp_gate_mul_kernel!(gate, up)
    i = @index(Global, Linear)
    @inbounds gate[i] = gate[i] / (one(eltype(gate)) + exp(-gate[i])) * up[i]
end

function fused_mlp_gate_mul!(gate::AbstractArray{T,1}, up::AbstractArray{T,1}) where T
    n = length(gate)
    n == 0 && return gate
    kfn = mlp_gate_mul_kernel!(_GPU_BACKEND)
    kfn(gate, up; ndrange=(n,))
    return gate
end

# ============================================================
# SiLU gate multiply: out = SiLU(x) .* gate (in-place on out)
# ============================================================
@kernel function silu_gate_mul_kernel!(out, x, gate)
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] / (one(eltype(x)) + exp(-x[i])) * gate[i]
end

function fused_silu_gate_mul!(out::AbstractArray{T,1}, x::AbstractArray{T,1}, gate::AbstractArray{T,1}) where T
    n = length(out)
    n == 0 && return out
    kfn = silu_gate_mul_kernel!(_GPU_BACKEND)
    kfn(out, x, gate; ndrange=(n,))
    return out
end

# ============================================================
# SSM gate sigmoid: out = sigmoid(bias + x) (in-place on out)
# Avoids allocating bias .+ x
# ============================================================
@kernel function ssm_gate_sigmoid_kernel!(out, bias, x)
    i = @index(Global, Linear)
    @inbounds out[i] = one(eltype(out)) / (one(eltype(out)) + exp(-(bias[i] + x[i])))
end

function fused_ssm_gate_sigmoid!(out::AbstractArray{T,1}, bias::AbstractArray{T,1}, x::AbstractArray{T,1}) where T
    n = length(out)
    n == 0 && return out
    kfn = ssm_gate_sigmoid_kernel!(_GPU_BACKEND)
    kfn(out, bias, x; ndrange=(n,))
    return out
end

# ============================================================
# SSM decay: out = exp(-a .* dt) (in-place on out)
# Avoids allocating exp.(-a .* dt)
# ============================================================
@kernel function ssm_decay_kernel!(out, a, dt)
    i = @index(Global, Linear)
    @inbounds out[i] = exp(-a[i] * dt[i])
end

function fused_ssm_decay!(out::AbstractArray{T,1}, a::AbstractArray{T,1}, dt::AbstractArray{T,1}) where T
    n = length(out)
    n == 0 && return out
    kfn = ssm_decay_kernel!(_GPU_BACKEND)
    kfn(out, a, dt; ndrange=(n,))
    return out
end

# ============================================================
# GPU-native argmax (single thread, O(n) — sufficient for vocab ~ 151k)
# Only copies a single Int back to CPU.
# ============================================================
@kernel function argmax_kernel!(logits, result)
    if @index(Global, Linear) == 1
        @inbounds result[1] = 1
        max_val = logits[1]
        @inbounds for i in 2:length(logits)
            if logits[i] > max_val
                max_val = logits[i]
                result[1] = i
            end
        end
    end
end

function gpu_argmax!(logits::AbstractArray{T,1}) where T
    result = oneAPI.oneArray{Int}(undef)
    kfn = argmax_kernel!(_GPU_BACKEND)
    kfn(logits, result; ndrange=(1,))
    return oneAPI.Array(result)[1]
end

# ============================================================
# GPU-native temperature + softmax (single thread for stability)
# Computes probability vector on GPU without CPU sync.
# ============================================================
@kernel function temperature_softmax_kernel!(logits, temperature, out_probs)
    n = length(logits)
    # Single-threaded for numerical stability
    @inbounds for i in 1:n
        out_probs[i] = logits[i] / temperature
    end
    # Find max for numerical stability
    max_val = out_probs[1]
    @inbounds for i in 2:n
        if out_probs[i] > max_val
            max_val = out_probs[i]
        end
    end
    # Compute exp and sum
    sum_exp = zero(eltype(logits))
    @inbounds for i in 1:n
        exp_val = exp(out_probs[i] - max_val)
        out_probs[i] = exp_val
        sum_exp += exp_val
    end
    # Normalize
    inv_sum = one(eltype(logits)) / sum_exp
    @inbounds for i in 1:n
        out_probs[i] *= inv_sum
    end
end

# ============================================================
# GPU-native search over cumulative probabilities
# Single thread, O(n), returns index of first element where cumsum >= r
# ============================================================
@kernel function gpu_search_kernel!(probs, r, result)
    if @index(Global, Linear) == 1
        n = length(probs)
        idx = n
        cumsum = zero(eltype(probs))
        @inbounds for i in 1:n
            cumsum += probs[i]
            if r <= cumsum && idx == n
                idx = i
            end
        end
        result[1] = idx
    end
end

# ============================================================
# GPU-native sampling — no CPU-GPU sync of probability vector!
# Only syncs a single Int back to CPU.
# ============================================================
function gpu_sample!(logits::AbstractArray{T,1}, temperature::S) where {T <: AbstractFloat, S <: AbstractFloat}
    n = length(logits)
    out_probs = oneAPI.oneArray{T}(undef, n)
    
    temp = convert(T, temperature)
    
    # GPU: temperature scaling + softmax (single thread for stability)
    softmax_kfn = temperature_softmax_kernel!(_GPU_BACKEND)
    softmax_kfn(logits, temp, out_probs; ndrange=(1,))
    
    # CPU: random number (only 4 bytes, negligible)
    r = rand(T)
    
    # GPU: binary search for sampled index
    result = oneAPI.oneArray{Int}(undef)
    search_kfn = gpu_search_kernel!(_GPU_BACKEND)
    search_kfn(out_probs, r, result; ndrange=(1,))
    
    # GPU->CPU: single Int (O(1) transfer)
    return oneAPI.Array(result)[1]
end

end # module
