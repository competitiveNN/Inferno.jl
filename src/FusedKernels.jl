module FusedKernels

using KernelAbstractions: @kernel, @index, @synchronize
using oneAPI
using LinearAlgebra

export fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!
export fused_ssm_gate_sigmoid!, fused_ssm_decay!, gpu_argmax!, gpu_sample!

# ============================================================
# L2 Norm (in-place) — avoids allocating x.^2
# Uses GPU dot for reduction then a 1-line scaling kernel.
# ============================================================
@kernel function l2norm_scale_kernel!(x, inv_norm::Float32)
    i = @index(Global, Linear)
    @inbounds x[i] = x[i] * inv_norm
end

function fused_l2norm!(x::AbstractArray{Float32,1}, eps::Float32)
    n = length(x)
    n == 0 && return x
    # GPU reduction via BLAS dot (no temp allocation, single kernel call)
    ss = dot(x, x)
    inv_norm = 1.0f0 / sqrt(ss + eps)
    kernel = l2norm_scale_kernel!
    kernel(x, inv_norm; ndrange=n)
    return x
end

# ============================================================
# Attention weighted sum across all heads (fused kernel)
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
        acc = 0.0f0
        for s in 1:seq_len
            acc += scores[h + 1, s] * v_cache[k_off + 1, s]
        end
        attn_out[idx] = acc
    end
end

function fused_attention_weighted_sum!(
    attn_out::AbstractArray{Float32,1},
    scores::AbstractArray{Float32,2},
    v_cache::AbstractArray{Float32,2},
    n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int
)
    n_total = n_heads * head_dim
    n_total == 0 && return attn_out
    kernel = attention_weighted_sum_kernel!
    kernel(attn_out, scores, v_cache, n_heads, head_dim, seq_len, n_groups; ndrange=n_total)
    return attn_out
end

# ============================================================
# MLP gate multiply: gate = SiLU(gate) .* up (in-place on gate)
# ============================================================
@kernel function mlp_gate_mul_kernel!(gate, up)
    i = @index(Global, Linear)
    @inbounds gate[i] = gate[i] / (1.0f0 + exp(-gate[i])) * up[i]
end

function fused_mlp_gate_mul!(gate::AbstractArray{Float32,1}, up::AbstractArray{Float32,1})
    n = length(gate)
    n == 0 && return gate
    kernel = mlp_gate_mul_kernel!
    kernel(gate, up; ndrange=n)
    return gate
end

# ============================================================
# SiLU gate multiply: out = SiLU(x) .* gate (in-place on out)
# ============================================================
@kernel function silu_gate_mul_kernel!(out, x, gate)
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] / (1.0f0 + exp(-x[i])) * gate[i]
end

function fused_silu_gate_mul!(out::AbstractArray{Float32,1}, x::AbstractArray{Float32,1}, gate::AbstractArray{Float32,1})
    n = length(out)
    n == 0 && return out
    kernel = silu_gate_mul_kernel!
    kernel(out, x, gate; ndrange=n)
    return out
end

# ============================================================
# SSM gate sigmoid: out = sigmoid(bias + x) (in-place on out)
# Avoids allocating bias .+ x
# ============================================================
@kernel function ssm_gate_sigmoid_kernel!(out, bias, x)
    i = @index(Global, Linear)
    @inbounds out[i] = 1.0f0 / (1.0f0 + exp(-(bias[i] + x[i])))
end

function fused_ssm_gate_sigmoid!(out::AbstractArray{Float32,1}, bias::AbstractArray{Float32,1}, x::AbstractArray{Float32,1})
    n = length(out)
    n == 0 && return out
    kernel = ssm_gate_sigmoid_kernel!
    kernel(out, bias, x; ndrange=n)
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

function fused_ssm_decay!(out::AbstractArray{Float32,1}, a::AbstractArray{Float32,1}, dt::AbstractArray{Float32,1})
    n = length(out)
    n == 0 && return out
    kernel = ssm_decay_kernel!
    kernel(out, a, dt; ndrange=n)
    return out
end

# ============================================================
# GPU-native argmax (returns index to CPU, no full array copy)
# Single reduction kernel: finds max and its index
# ============================================================
@kernel function argmax_kernel!(logits, result)
    # Simple 1-thread reduction (sufficient for vocab ~ 151k)
    # For larger vocab, use parallel reduction
    @inbounds result[1] = 1
    max_val = logits[1]
    @inbounds for i in 2:length(logits)
        if logits[i] > max_val
            max_val = logits[i]
            result[1] = i
        end
    end
end

function gpu_argmax!(logits::AbstractArray{Float32,1})
    result = oneAPI.oneArray{Int}(undef, 1)
    kernel = argmax_kernel!
    kernel(logits, result; ndrange=1)
    # Must sync here to get result back to CPU for next token ID
    # But this is O(1) transfer vs O(vocab) for full Array(logits)
    return oneAPI.Array(result)[1]
end

# ============================================================
# GPU-native temperature + top-k sampling
# Returns sampled token index to CPU (O(1) transfer)
# ============================================================
@kernel function temperature_topk_kernel!(logits, temperature, out_probs)
    n = length(logits)
    # Step 1: Apply temperature scaling in-place
    @inbounds for i in 1:n
        logits[i] = logits[i] / temperature
    end
    # Step 2: Find max for numerical stability
    max_val = logits[1]
    @inbounds for i in 2:n
        if logits[i] > max_val
            max_val = logits[i]
        end
    end
    # Step 3: Compute exp and sum
    sum_exp = 0.0f0
    @inbounds for i in 1:n
        exp_val = exp(logits[i] - max_val)
        out_probs[i] = exp_val
        sum_exp += exp_val
    end
    # Step 4: Normalize
    inv_sum = 1.0f0 / sum_exp
    @inbounds for i in 1:n
        out_probs[i] *= inv_sum
    end
end

function gpu_sample!(logits::AbstractArray{Float32,1}, temperature::Float32)
    n = length(logits)
    out_probs = oneAPI.oneArray{Float32}(undef, n)
    
    kernel = temperature_topk_kernel!
    kernel(logits, temperature, out_probs; ndrange=1)
    
    # Copy probs to CPU for sampling (O(vocab) - unavoidable for now)
    # In future: implement GPU-side cumulative sum + binary search
    cpu_probs = oneAPI.Array(out_probs)
    
    # CPU-side sample
    r = rand(Float32)
    cumsum = 0.0f0
    @inbounds for i in 1:n
        cumsum += cpu_probs[i]
        if r <= cumsum
            return i
        end
    end
    return n
end

end # module
