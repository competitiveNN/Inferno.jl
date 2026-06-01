"""
CommonOps — Shared CPU inference operations for all model backends.

Provides common primitives used by both Qwen3.5 (ModelCPU) and Gemma4:
- RMSNorm (with and without weight scaling)
- Repetition penalty
- Sampling (softmax-based with temperature, top-k, top-p, min-p)
- Logit softcapping

Each model backend imports and re-uses these rather than defining its own copies.
"""
module CommonOps

export rmsnorm!, rmsnorm_no_scale!, apply_repetition_penalty!, softmax_sample, logit_softcap!

"""
    rmsnorm!(out, x, w, eps)

In-place RMSNorm: out[i] = x[i] * inv_rms * w[i]
where inv_rms = 1 / sqrt(mean(x²) + eps)

This is the standard RMSNorm used by both Gemma4 and Qwen3.5 (with raw weights, no +1).
"""
function rmsnorm!(out::AbstractVector{Float32}, x::AbstractVector{Float32},
                  w::AbstractVector{Float32}, eps::Float32)
    n = length(x)
    ss = 0.0f0
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n + eps)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms * w[i]
    end
end

"""
    rmsnorm_no_scale!(out, x, eps)

In-place RMSNorm without weight scaling: out[i] = x[i] * inv_rms
Used by Gemma4 attention for Q/K normalization.
"""
function rmsnorm_no_scale!(out::AbstractVector{Float32}, x::AbstractVector{Float32}, eps::Float32)
    n = length(x)
    ss = 0.0f0
    @simd for i in 1:n
        ss += x[i] * x[i]
    end
    inv_rms = 1.0f0 / sqrt(ss / n + eps)
    @simd for i in 1:n
        out[i] = x[i] * inv_rms
    end
end

"""
    apply_repetition_penalty!(logits, token_counts, penalty)

Apply multiplicative repetition penalty to logits for previously-seen tokens.
- Positive logits are divided by penalty
- Negative logits are multiplied by penalty
- penalty=1.0 is a no-op (skipped)
"""
function apply_repetition_penalty!(logits::Vector{Float32}, token_counts::Dict{Int,Int}, penalty::Float32)
    if penalty == 1.0f0
        return
    end
    for (tid, count) in token_counts
        if tid >= 1 && tid <= length(logits)
            if logits[tid] > 0
                logits[tid] /= penalty
            else
                logits[tid] *= penalty
            end
        end
    end
end

"""
    softmax_sample(logits; temperature, top_p, top_k, min_p)

Sample a token from logits using temperature, top-k, top-p, and min-p filtering.

# Arguments
- `logits::Vector{Float32}`: Raw model logits
- `temperature::Float32=1.0f0`: Sampling temperature (0 = greedy/argmax)
- `top_p::Float32=1.0f0`: Nucleus sampling threshold (1.0 = disabled)
- `top_k::Int=0`: Top-k filtering (0 = disabled)
- `min_p::Float32=0.0f0`: Minimum probability threshold (0.0 = disabled)

# Returns
- `Int`: Sampled token ID (1-indexed)
"""
function softmax_sample(logits::Vector{Float32}; temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0, min_p::Float32=0.0f0)
    # Handle temperature=0 (greedy/argmax sampling)
    if temperature == 0.0f0
        return argmax(logits)
    end
    
    # Apply temperature
    if temperature != 1.0f0
        logits = logits ./ temperature
    end
    
    # Apply top-k filtering using partialsortperm (O(N log k) instead of O(N log N))
    if top_k > 0 && top_k < length(logits)
        # Get top-k indices without full sort
        k = min(top_k, length(logits))
        top_k_indices = partialsortperm(logits, 1:k, rev=true)
        
        # Create a boolean mask instead of Set (avoids allocation)
        keep_mask = falses(length(logits))
        @simd for idx in top_k_indices
            keep_mask[idx] = true
        end
        
        # Zero out non-top-k logits
        @simd for i in 1:length(logits)
            if !keep_mask[i]
                logits[i] = -Inf32
            end
        end
    elseif top_k > 0
        # top_k >= length(logits), no filtering needed
    end
    
    # Apply softmax
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0f0
        # Get sorted indices by probability
        sorted_indices = sortperm(probs, rev=true)
        
        # Find cumulative sum threshold without Set allocation
        cumsum = 0.0f0
        keep_count = 0
        for (i, idx) in enumerate(sorted_indices)
            cumsum += probs[idx]
            keep_count = i
            if cumsum >= top_p
                break
            end
        end
        
        # Create boolean mask for kept indices
        keep_mask = falses(length(probs))
        for i in 1:keep_count
            keep_mask[sorted_indices[i]] = true
        end
        
        # Zero out probabilities for tokens not in top-p
        @simd for i in 1:length(probs)
            if !keep_mask[i]
                probs[i] = 0.0f0
            end
        end
        
        # Renormalize
        total = sum(probs)
        if total > 0.0f0
            probs ./= total
        end
    end
    
    # Apply minimum probability threshold (relative to max probability)
    if min_p > 0.0f0
        max_prob = maximum(probs)
        threshold = max_prob * min_p
        @simd for i in 1:length(probs)
            if probs[i] < threshold
                probs[i] = 0.0f0
            end
        end
        # Renormalize
        total = sum(probs)
        if total > 0.0f0
            probs ./= total
        end
    end
    
    # Sample from the distribution
    r = rand(Float32)
    cumsum = 0.0f0
    @simd for i in 1:length(probs)
        cumsum += probs[i]
        if r <= cumsum
            return i
        end
    end
    
    return length(probs)
end

"""
    logit_softcap!(logits, cap)

Apply logit softcapping: logits[i] = tanh(logits[i] / cap) * cap
Used by Gemma4 final logits. No-op when cap <= 0.
"""
function logit_softcap!(logits::Vector{Float32}, cap::Float32)
    if cap <= 0.0f0
        return
    end
    @simd for i in 1:length(logits)
        logits[i] = tanh(logits[i] / cap) * cap
    end
end

end # module CommonOps
