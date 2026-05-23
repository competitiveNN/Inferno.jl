module FusedKernels

using KernelAbstractions: @kernel, @index, @synchronize
using oneAPI
using LinearAlgebra

export fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!
export fused_ssm_gate_sigmoid!, fused_ssm_decay!

# ============================================================
# L2 Norm (in-place) — avoids allocating x.^2
# Uses GPU dot for reduction then a 1-line scaling kernel.
# ============================================================
@kernel function l2norm_scale_kernel!(x, inv_norm::Float32)
    i = @index(Global, Linear)
    @inbounds x[i] = x[i] * inv_norm
end

function fused_l2norm!(x::oneAPI.oneArray{Float32,1}, eps::Float32)
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
    attn_out::oneAPI.oneArray{Float32,1},
    scores::oneAPI.oneArray{Float32,2},
    v_cache::oneAPI.oneArray{Float32,2},
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

function fused_mlp_gate_mul!(gate::oneAPI.oneArray{Float32,1}, up::oneAPI.oneArray{Float32,1})
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

function fused_silu_gate_mul!(out::oneAPI.oneArray{Float32,1}, x::oneAPI.oneArray{Float32,1}, gate::oneAPI.oneArray{Float32,1})
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

function fused_ssm_gate_sigmoid!(out::oneAPI.oneArray{Float32,1}, bias::oneAPI.oneArray{Float32,1}, x::oneAPI.oneArray{Float32,1})
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

function fused_ssm_decay!(out::oneAPI.oneArray{Float32,1}, a::oneAPI.oneArray{Float32,1}, dt::oneAPI.oneArray{Float32,1})
    n = length(out)
    n == 0 && return out
    kernel = ssm_decay_kernel!
    kernel(out, a, dt; ndrange=n)
    return out
end

end # module
