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

export rmsnorm!, rmsnorm_no_scale!, apply_repetition_penalty!, softmax_sample, softmax_sample_scratch!, create_softmax_scratch, SoftmaxScratch, logit_softcap!

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

struct SoftmaxScratch
    logits_buf::Vector{Float32}
    exp_probs::Vector{Float32}
    indices::Vector{Int}
    keep_mask::Vector{Bool}
end

@inline function _heap_value_less(values::Vector{Float32}, indices::Vector{Int}, i::Int, j::Int)
    return values[indices[i]] < values[indices[j]]
end

@inline function _heap_sift_down!(values::Vector{Float32}, indices::Vector{Int}, root::Int, heap_size::Int)
    while true
        left = 2 * root
        right = left + 1
        largest = root
        if left <= heap_size && _heap_value_less(values, indices, largest, left)
            largest = left
        end
        if right <= heap_size && _heap_value_less(values, indices, largest, right)
            largest = right
        end
        if largest == root
            return nothing
        end
        indices[root], indices[largest] = indices[largest], indices[root]
        root = largest
    end
end

@inline function _heapify!(values::Vector{Float32}, indices::Vector{Int}, n::Int)
    for root in (n ÷ 2):-1:1
        _heap_sift_down!(values, indices, root, n)
    end
    return nothing
end

@inline function _heap_pop!(values::Vector{Float32}, indices::Vector{Int}, heap_size::Int)
    result = indices[1]
    indices[1] = indices[heap_size]
    _heap_sift_down!(values, indices, 1, heap_size - 1)
    return result
end

function create_softmax_scratch(n::Int)
    return SoftmaxScratch(Vector{Float32}(undef, n), Vector{Float32}(undef, n), Vector{Int}(undef, n), Vector{Bool}(undef, n))
end

@inline function _validate_sampling_parameters(temperature::Float32, top_p::Float32, top_k::Int, min_p::Float32)
    if !isfinite(temperature) || temperature < 0.0f0
        throw(ArgumentError("temperature must be finite and non-negative"))
    end
    if !isfinite(top_p) || top_p < 0.0f0 || top_p > 1.0f0
        throw(ArgumentError("top_p must be in [0, 1]"))
    end
    if top_k < 0
        throw(ArgumentError("top_k must be non-negative"))
    end
    if !isfinite(min_p) || min_p < 0.0f0 || min_p > 1.0f0
        throw(ArgumentError("min_p must be in [0, 1]"))
    end
    return nothing
end

@inline function _validate_logits(logits::Vector{Float32})
    max_logit = -Inf32
    argmax_index = 1
    all_negative_infinite = true

    @inbounds for i in eachindex(logits)
        logit = logits[i]
        if isnan(logit)
            throw(ArgumentError("logits must not contain NaN"))
        elseif logit == Inf32
            throw(ArgumentError("logits must not contain +Inf"))
        elseif logit != -Inf32
            all_negative_infinite = false
            if logit > max_logit
                max_logit = logit
                argmax_index = i
            end
        end
    end

    return argmax_index, all_negative_infinite
end

"""
    softmax_sample(logits; temperature, top_p, top_k, min_p)

Sample a token from logits using temperature, top-k, top-p, and min-p filtering.
This wrapper creates scratch space for callers that do not maintain a reusable
`SoftmaxScratch`.
"""
function softmax_sample(logits::Vector{Float32}; temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0, min_p::Float32=0.0f0)
    _validate_sampling_parameters(temperature, top_p, top_k, min_p)

    n = length(logits)
    n == 0 && throw(ArgumentError("logits must not be empty"))

    greedy_index, all_negative_infinite = _validate_logits(logits)
    if temperature == 0.0f0 || n == 1 || all_negative_infinite
        return greedy_index
    end

    return softmax_sample_scratch!(logits, create_softmax_scratch(n);
        temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
end

"""
    softmax_sample_scratch!(logits, scratch; temperature, top_p, top_k, min_p)

Sample from logits while reusing scratch buffers. The input logits are preserved.
Only the first `length(logits)` elements of each scratch buffer are initialized.

`-Inf` logits are treated as impossible. If every logit is `-Inf`, the first
token is returned deterministically. `+Inf` and `NaN` logits are rejected.
"""
function softmax_sample_scratch!(logits::Vector{Float32}, scratch::SoftmaxScratch; temperature::Float32=1.0f0, top_p::Float32=1.0f0, top_k::Int=0, min_p::Float32=0.0f0)
    _validate_sampling_parameters(temperature, top_p, top_k, min_p)

    n = length(logits)
    n == 0 && throw(ArgumentError("logits must not be empty"))
    length(scratch.logits_buf) < n && throw(ArgumentError("softmax scratch buffer is too small"))
    length(scratch.exp_probs) < n && throw(ArgumentError("softmax probability buffer is too small"))
    length(scratch.indices) < n && throw(ArgumentError("softmax index buffer is too small"))
    length(scratch.keep_mask) < n && throw(ArgumentError("softmax mask buffer is too small"))

    greedy_index, all_negative_infinite = _validate_logits(logits)
    if temperature == 0.0f0 || n == 1 || all_negative_infinite
        return greedy_index
    end

    @inbounds for i in 1:n
        scratch.indices[i] = i
        scratch.keep_mask[i] = false
        scratch.logits_buf[i] = logits[i]
    end

    if top_k > 0 && top_k < n
        _heapify!(scratch.logits_buf, scratch.indices, n)
        heap_size = n
        for _ in 1:top_k
            idx = _heap_pop!(scratch.logits_buf, scratch.indices, heap_size)
            heap_size -= 1
            scratch.keep_mask[idx] = true
        end
        @simd for i in 1:n
            if !scratch.keep_mask[i]
                scratch.logits_buf[i] = -Inf32
            end
        end
    end

    # Stable softmax: divide logit differences by temperature without scaling
    # the raw logits themselves.
    max_logit = -Inf32
    @simd for i in 1:n
        max_logit = max(max_logit, scratch.logits_buf[i])
    end
    total = 0.0f0
    @simd for i in 1:n
        p = exp((scratch.logits_buf[i] - max_logit) / temperature)
        scratch.exp_probs[i] = p
        total += p
    end
    inv_total = 1.0f0 / total
    @simd for i in 1:n
        scratch.exp_probs[i] *= inv_total
    end

    if top_p < 1.0f0
        _heapify!(scratch.exp_probs, scratch.indices, n)
        cumsum = 0.0f0
        heap_size = n
        @inbounds for i in 1:n
            scratch.keep_mask[i] = false
        end
        while heap_size > 0
            idx = _heap_pop!(scratch.exp_probs, scratch.indices, heap_size)
            heap_size -= 1
            scratch.keep_mask[idx] = true
            cumsum += scratch.exp_probs[idx]
            cumsum >= top_p && break
        end
        @simd for i in 1:n
            if !scratch.keep_mask[i]
                scratch.exp_probs[i] = 0.0f0
            end
        end
        total = 0.0f0
        @simd for i in 1:n
            total += scratch.exp_probs[i]
        end
        if total > 0.0f0
            inv_total = 1.0f0 / total
            @simd for i in 1:n
                scratch.exp_probs[i] *= inv_total
            end
        end
    end

    if min_p > 0.0f0
        max_prob = 0.0f0
        @simd for i in 1:n
            max_prob = max(max_prob, scratch.exp_probs[i])
        end
        threshold = max_prob * min_p
        @simd for i in 1:n
            if scratch.exp_probs[i] < threshold
                scratch.exp_probs[i] = 0.0f0
            end
        end
        total = 0.0f0
        @simd for i in 1:n
            total += scratch.exp_probs[i]
        end
        if total > 0.0f0
            inv_total = 1.0f0 / total
            @simd for i in 1:n
                scratch.exp_probs[i] *= inv_total
            end
        end
    end

    r = rand(Float32)
    cumsum = 0.0f0
    last_positive_index = greedy_index
    @inbounds for i in 1:n
        p = scratch.exp_probs[i]
        if p > 0.0f0
            last_positive_index = i
        end
        cumsum += p
        if r <= cumsum
            return i
        end
    end

    return last_positive_index
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
