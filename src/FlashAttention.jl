# --- Flash Attention Implementation (CPU Optimized) ---
# This is included inside ModelCPU module

"""
    flash_attention_cpu!(output, Q, cache_k, cache_v, kv_h, seq_len, scale, head_dim)

Memory-efficient attention that avoids materializing the full attention matrix.

Key optimizations:
1. Tiled computation - process attention in blocks to fit in cache
2. Online softmax - compute softmax incrementally without full materialization
3. Recomputation of attention scores from KV cache instead of storing
4. Pre-allocated scores buffer - avoid per-block allocations
5. Fused block max - track maximum during score computation
6. @turbo vectorization - use LoopVectorization for SIMD

This is essentially Flash Attention-2/3 adapted for CPU.
"""
function flash_attention_cpu!(
 output::AbstractVector{Float32},
 Q::AbstractVector{Float32},
 cache_k::AbstractArray{Float32,3},
 cache_v::AbstractArray{Float32,3},
 kv_h::Int,
 seq_len::Int,
    scale::Float32,
    head_dim::Int
)
    fill!(output, 0.0f0)
    
    BLOCK_N = 64  # Key/Value block size
    
    # Pre-allocate scores buffer to avoid per-block allocation
    scores = Vector{Float32}(undef, BLOCK_N)
    
 # Online softmax state
 m = -Inf32
 l = 0.0f0
 
 # Process KV cache in blocks
 for j in 1:BLOCK_N:seq_len
     j_end = min(j + BLOCK_N - 1, seq_len)
     block_len = j_end - j + 1
     
     # Compute scores with fused max tracking
     m_new = m
     for n in 1:block_len
         k_j = j + n - 1
         s = 0.0f0
         @turbo for d in 1:head_dim
             s += Q[d] * cache_k[d, kv_h, k_j]
         end
         score = s * scale
            scores[n] = score
            if score > m_new
             m_new = score
         end
        end
     
     scale_factor = exp(m - m_new)
     
     if m_new > m
         output .*= scale_factor
            l *= scale_factor
        end
        
        for n in 1:block_len
         k_j = j + n - 1
         p = exp(scores[n] - m_new)
         l += p
         @turbo for d in 1:head_dim
             output[d] += p * cache_v[d, kv_h, k_j]
         end
     end
     
     m = m_new
 end
 
 inv_l = 1.0f0 / l
 @turbo for d in 1:head_dim
     output[d] *= inv_l
 end
 return output
end
