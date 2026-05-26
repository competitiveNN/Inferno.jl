"""
GPUCommon — Unified KernelAbstractions.jl kernels for Inferno.jl

Provides fused GPU kernels for all model backends (Gemma4, Qwen3.5).
All kernels use KernelAbstractions.jl for portability and performance.
"""
module GPUCommon

using KernelAbstractions: @kernel, @index, @localmem, @synchronize
using LinearAlgebra
using oneAPI

export rmsnorm_kernel!, gelu_kernel!, residual_add_kernel!, apply_rope_kernel!
export softmax_kernel!, matmul_vec_kernel!, silu_kernel!, sigmoid_kernel!
export batched_attention_scores_kernel!, batched_softmax_kernel!, batched_ssm_state_kernel!
export rmsnorm_gpu!, gelu_gpu!, residual_add_gpu!, apply_rope_gpu!, softmax_gpu!, silu_gpu!, sigmoid_gpu!
export batched_attention_scores!, batched_softmax!, batched_ssm_state_update!, batched_ssm_output_sum!, fused_attention_forward!
export write_kv_cache_kernel!, write_kv_cache_gpu!

# ============================================================
# RMSNorm Kernel
# ============================================================
"""
 rmsnorm_kernel!(out, x, weight, eps, sum_sq)

In-place RMSNorm: out[i] = x[i] * inv_rms * weight[i]
where inv_rms = 1 / sqrt(sum_sq / n + eps)

The sum_sq must be pre-computed (reduction step) to avoid O(n²) work.
"""
@kernel function rmsnorm_kernel!(out, x, weight, eps::Float32, sum_sq::Float32)
 i = @index(Global, Linear)
 n = length(x)
 
 inv_rms = 1.0f0 / sqrt(sum_sq / n + eps)
 
 @inbounds out[i] = x[i] * inv_rms * weight[i]
end

function rmsnorm_gpu!(out::AbstractArray, x::AbstractArray, weight::AbstractArray, eps::Float32)
 n = length(x)
 if n == 0 return out end
 
 # Step 1: Compute sum of squares (GPU-accelerated BLAS dot product)
    sum_sq = dot(x, x)
 
 # Step 2: Normalize using the pre-computed sum
 kernel = rmsnorm_kernel!
 kernel(out, x, weight, eps, sum_sq; ndrange=n)
 @synchronize()
 return out
end

# ============================================================
# GELU Kernel (with tanh approximation)
# ============================================================
@inline function gelu_tanh(x)
    cdf = 0.5f0 * (1.0f0 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))
    return 0.5f0 * x * (1.0f0 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x^3)))
end

@kernel function gelu_kernel!(out, x)
    i = @index(Global, Linear)
    @inbounds out[i] = gelu_tanh(x[i])
end

function gelu_gpu!(out::AbstractArray, x::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kernel = gelu_kernel!
    kernel(out, x; ndrange=n)
    @synchronize()
    return out
end

# ============================================================
# Residual Add Kernel
# ============================================================
@kernel function residual_add_kernel!(out, x, residual)
    i = @index(Global, Linear)
    @inbounds out[i] = x[i] + residual[i]
end

function residual_add_gpu!(out::AbstractArray, x::AbstractArray, residual::AbstractArray)
    n = length(x)
    if n == 0 return out end
    kernel = residual_add_kernel!
    kernel(out, x, residual; ndrange=n)
    @synchronize()
    return out
end

# ============================================================
# Rotary Embedding Kernel
# ============================================================
@kernel function apply_rope_kernel!(x, cos, sin, rotary_dim)
    i = @index(Global, Linear)
    # i corresponds to (head, dim) flattened
    # We assume x is (head_dim, num_heads) flattened to 1D
    # This is a simplified version; production needs proper indexing
    
    # For now, assume x is (rotary_dim, num_heads) and we process per head
    # This kernel is a placeholder for the full RoPE implementation
    # which requires 2D indexing (head, dim)
end

function apply_rope_gpu!(x::AbstractArray, cos::AbstractArray, sin::AbstractArray, rotary_dim::Int)
    # Placeholder: Implement proper 2D indexing for RoPE
    # This requires knowing the exact layout of x
    n = length(x)
    if n == 0 return x end
    kernel = apply_rope_kernel!
    kernel(x, cos, sin, rotary_dim; ndrange=n)
    @synchronize()
    return x
end

# ============================================================
# Softmax Kernel
# ============================================================
@kernel function softmax_kernel!(out, x, temperature::Float32)
    i = @index(Global, Linear)
    n = length(x)
    
    # Find max for stability
    max_val = -Inf32
    @inbounds for j in 1:n
        max_val = max(max_val, x[j])
    end
    
    # Compute exp and sum
    sum_exp = 0.0f0
    @inbounds for j in 1:n
        sum_exp += exp((x[j] - max_val) / temperature)
    end
    
    @inbounds out[i] = exp((x[i] - max_val) / temperature) / sum_exp
end

function softmax_gpu!(out::AbstractArray, x::AbstractArray, temperature::Float32 = 1.0f0)
 n = length(x)
 if n == 0 return out end
 kernel = softmax_kernel!
 kernel(out, x, temperature; ndrange=n)
 @synchronize()
 return out
end

# ============================================================
# SiLU Kernel (Swish: x * sigmoid(x))
# ============================================================
@kernel function silu_kernel!(out, x)
 i = @index(Global, Linear)
 @inbounds out[i] = x[i] / (1.0f0 + exp(-x[i]))
end

function silu_gpu!(out::AbstractArray, x::AbstractArray)
 n = length(x)
 if n == 0 return out end
 kernel = silu_kernel!
 kernel(out, x; ndrange=n)
 @synchronize()
 return out
end

# ============================================================
# Sigmoid Kernel
# ============================================================
@kernel function sigmoid_kernel!(out, x)
 i = @index(Global, Linear)
 @inbounds out[i] = 1.0f0 / (1.0f0 + exp(-x[i]))
end

function sigmoid_gpu!(out::AbstractArray, x::AbstractArray)
 n = length(x)
 if n == 0 return out end
 kernel = sigmoid_kernel!
 kernel(out, x; ndrange=n)
 @synchronize()
 return out
end

# ============================================================
# Batched Attention Scores Kernel (No CPU loops!)
# Computes scores[h, s] = Q[h] · K[kv_group(h), :, s] / sqrt(head_dim)
# for ALL heads and positions in parallel
# ============================================================
"""
 batched_attention_scores_kernel!(scores, q, k_cache, n_heads, head_dim, seq_len, n_groups)

Compute attention scores for ALL heads and positions in parallel.
- scores: (n_heads, seq_len) output buffer
- q: (n_heads * head_dim) query vector
- k_cache: (n_kv_heads * head_dim, seq_len) key cache
- n_heads: total number of attention heads
- head_dim: dimension per head
- seq_len: current sequence length
- n_groups: number of KV groups (n_heads / n_kv_heads)
"""
@kernel function batched_attention_scores_kernel!(scores, q, k_cache, n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
 idx = @index(Global, Linear)
 n_total = n_heads * seq_len
 
 # Use conditional instead of return
 if idx <= n_total
  h = (idx - 1) ÷ seq_len  # 0-based head index
  s = (idx - 1) % seq_len  # 0-based position
  
  # Find KV group for this head
  kv_h = h ÷ n_groups
  
  # Compute dot product: Q[h] · K[kv_h, :, s]
  q_off = h * head_dim
  k_off = kv_h * head_dim
  k_pos_off = s  # column index in k_cache (0-based, but Julia is 1-based)
  
  dot_sum = 0.0f0
  for d in 0:(head_dim-1)
   q_val = q[q_off + d + 1]
   k_val = k_cache[k_off + d + 1, k_pos_off + 1]
   dot_sum += q_val * k_val
  end
  
  # Store score scaled by 1/sqrt(head_dim)
  scores[idx] = dot_sum / sqrt(Float32(head_dim))
 end
end

function batched_attention_scores!(scores::AbstractArray{Float32,2}, q::AbstractArray{Float32,1}, 
                                   k_cache::AbstractArray{Float32,2}, n_heads::Int, head_dim::Int, 
                                   seq_len::Int, n_groups::Int)
 n_total = n_heads * seq_len
 kernel = batched_attention_scores_kernel!
 kernel(scores, q, k_cache, n_heads, head_dim, seq_len, n_groups; ndrange=n_total)
 @synchronize()
 return scores
end

# ============================================================
# Batched Softmax Kernel (No CPU loops!)
# Computes softmax for ALL heads in parallel
# ============================================================
"""
 batched_softmax_kernel!(scores, n_heads, seq_len)

Apply softmax to ALL heads in parallel.
- scores: (n_heads, seq_len) input/output buffer
- n_heads: number of heads
- seq_len: sequence length

Each thread handles one (head, pos) pair.
Uses online softmax computation with per-row max/sum.
"""
@kernel function batched_softmax_kernel!(scores, n_heads::Int, seq_len::Int)
 idx = @index(Global, Linear)
 n_total = n_heads * seq_len
 
 if idx <= n_total
  h = (idx - 1) ÷ seq_len  # 0-based head index
  s = (idx - 1) % seq_len  # 0-based position
  row_start = h * seq_len + 1
  
 # scores is stored in column-major format
 # For head h (0-based), position j (0-based), the correct linear index is:
 # j * n_heads + h + 1
 
 # Step 1: Find max in this row
 max_val = -Inf32
 for j in 0:(seq_len-1)
  val = scores[j * n_heads + h + 1]  # Corrected indexing
  if val > max_val
   max_val = val
  end
 end
 
 # Step 2: Compute exp and sum
 exp_val = exp(scores[idx] - max_val)
 
 # Step 3: Compute sum
 sum_exp = 0.0f0
 for j in 0:(seq_len-1)
  sum_exp += exp(scores[j * n_heads + h + 1] - max_val)  # Corrected indexing
 end
 
 # Step 4: Normalize
 scores[idx] = exp_val / sum_exp
 end
end

function batched_softmax!(scores::AbstractArray{Float32,2}, n_heads::Int, seq_len::Int)
 n_total = n_heads * seq_len
 kernel = batched_softmax_kernel!
 kernel(scores, n_heads, seq_len; ndrange=n_total)
 @synchronize()
 return scores
end

# ============================================================
# Batched SSM State Update Kernel (No CPU loops!)
# Updates all SSM states in parallel
# ============================================================
"""
 batched_ssm_state_kernel!(h_state, decay, beta, x_conv, n_v, head_v, head_k)

Update SSM hidden states for ALL v_heads in parallel.
- h_state: (head_v, head_k, n_v) state tensor
- decay: (n_v) decay factors
- beta: (n_v * head_v) beta weights
- x_conv: (n_v * head_v) conv output
- n_v: number of v_heads
- head_v: dimension per v_head
- head_k: dimension per k_head (for outer product)

State update: h[t] = decay * h[t-1] + beta * x^T
"""
@kernel function batched_ssm_state_kernel!(h_state, decay, beta, x_conv, n_v::Int, head_v::Int, head_k::Int)
 idx = @index(Global, Linear)
 n_total = n_v * head_v * head_k
 
 if idx <= n_total
  # Flatten index to (v_head, h_dim, k_dim)
  v_head = (idx - 1) ÷ (head_v * head_k)
  rem = (idx - 1) % (head_v * head_k)
  h_dim = rem ÷ head_k
  k_dim = rem % head_k
  
  vb = v_head * head_v
  
  # Get values
  decay_val = decay[v_head + 1]
  beta_val = beta[vb + h_dim + 1]
  x_val = x_conv[vb + h_dim + 1]
  
  # Previous state
  old_state = h_state[h_dim + 1, k_dim + 1, v_head + 1]
  
  # State update
  new_state = decay_val * old_state + beta_val * x_val
  
  h_state[h_dim + 1, k_dim + 1, v_head + 1] = new_state
 end
end

function batched_ssm_state_update!(h_state::AbstractArray{Float32,3}, decay::AbstractArray{Float32,1},
                                   beta::AbstractArray{Float32,1}, x_conv::AbstractArray{Float32,1},
                                   n_v::Int, head_v::Int, head_k::Int)
 n_total = n_v * head_v * head_k
 kernel = batched_ssm_state_kernel!
 kernel(h_state, decay, beta, x_conv, n_v, head_v, head_k; ndrange=n_total)
 @synchronize()
 return h_state
end

# ============================================================
# Batched SSM Output Sum Kernel (No CPU loops!)
# Sums h_state across the k_dim dimension for all v_heads
# ============================================================
"""
 batched_ssm_output_kernel!(y_out, h_state, n_v, head_v, head_k)

Sum h_state over k_dim for each v_head and output to y_out.
- y_out: (n_v * head_v) output buffer
- h_state: (head_v, head_k, n_v) state tensor
"""
@kernel function batched_ssm_output_kernel!(y_out, h_state, n_v::Int, head_v::Int, head_k::Int)
 idx = @index(Global, Linear)
 n_total = n_v * head_v
 
 if idx <= n_total
  v_head = (idx - 1) ÷ head_v
  h_dim = (idx - 1) % head_v
  
  # Sum over k_dim
  sum_val = 0.0f0
  for k in 0:(head_k-1)
   sum_val += h_state[h_dim + 1, k + 1, v_head + 1]
  end
  
  y_out[idx] = sum_val
 end
end

function batched_ssm_output_sum!(y_out::AbstractArray{Float32,1}, h_state::AbstractArray{Float32,3},
                                 n_v::Int, head_v::Int, head_k::Int)
 n_total = n_v * head_v
 kernel = batched_ssm_output_kernel!
 kernel(y_out, h_state, n_v, head_v, head_k; ndrange=n_total)
 @synchronize()
 return y_out
end

# ============================================================
# KV Cache Write Kernel (Replaces CPU loop over kv_heads)
# Single kernel dispatch for all heads
# ============================================================
@kernel function write_kv_cache_kernel!(k_cache, v_cache, k, v, head_dim::Int, pos::Int)
    idx = @index(Global, Linear)
    n = length(k)
    if idx <= n
        h = (idx - 1) ÷ head_dim  # 0-based head index
        d = (idx - 1) % head_dim  # 0-based dim index
        # Write directly - k_cache is (head_dim, pos+1) at this head
        @inbounds k_cache[idx, pos+1] = k[idx]
        @inbounds v_cache[idx, pos+1] = v[idx]
    end
end

function write_kv_cache_gpu!(k_cache, v_cache, k, v, head_dim::Int, pos::Int)
    n = length(k)
    n == 0 && return nothing
    kernel = write_kv_cache_kernel!
    kernel(k_cache, v_cache, k, v, head_dim, pos; ndrange=n)
    @synchronize()
    return nothing
end

# ============================================================
# Fused Attention Kernel (scores + softmax + weighted sum)
# Combines the three attention steps into a single kernel launch.
# Each thread handles one (head, dim) pair.
# Recomputes scores 3x per thread to avoid intermediate buffer.
# ============================================================
@kernel function fused_attention_forward_kernel!(attn_out, q, k_cache, v_cache, n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * head_dim
    if idx <= n_total
        h = (idx - 1) ÷ head_dim  # 0-based head index
        d = (idx - 1) % head_dim  # 0-based dim index
        kv_h = h ÷ n_groups
        q_off = h * head_dim
        k_off = kv_h * head_dim
        v_off = kv_h * head_dim
        
        # Phase 1: Compute all scores for this head and track max
        max_val = -Inf32
        for s in 0:(seq_len-1)
            dot_sum = 0.0f0
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(Float32(head_dim))
            if score > max_val
                max_val = score
            end
        end
        
        # Phase 2: Compute sum of exp(score - max_val)
        sum_exp = 0.0f0
        for s in 0:(seq_len-1)
            # Recompute score for this position
            dot_sum = 0.0f0
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(Float32(head_dim))
            sum_exp += exp(score - max_val)
        end
        
        # Phase 3: Compute weighted sum for this dimension
        acc = 0.0f0
        for s in 0:(seq_len-1)
            # Recompute score for this position
            dot_sum = 0.0f0
            for dim in 0:(head_dim-1)
                q_val = q[q_off + dim + 1]
                k_val = k_cache[k_off + dim + 1, s + 1]
                dot_sum += q_val * k_val
            end
            score = dot_sum / sqrt(Float32(head_dim))
            prob = exp(score - max_val) / sum_exp
            v_val = v_cache[v_off + d + 1, s + 1]
            acc += prob * v_val
        end
        
        attn_out[idx] = acc
    end
end

function fused_attention_forward!(attn_out::AbstractArray{Float32,1}, q::AbstractArray{Float32,1}, 
                                  k_cache::AbstractArray{Float32,2}, v_cache::AbstractArray{Float32,2},
                                  n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int)
    n_total = n_heads * head_dim
    if n_total == 0
        return attn_out
    end
    kernel = fused_attention_forward_kernel!
    kernel(attn_out, q, k_cache, v_cache, n_heads, head_dim, seq_len, n_groups; ndrange=n_total)
    @synchronize()
    return attn_out
end

end # module GPUCommon
