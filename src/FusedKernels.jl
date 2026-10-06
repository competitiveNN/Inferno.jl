module FusedKernels

using KernelAbstractions: @kernel, @index, @synchronize
using oneAPI
using oneAPI: oneAPIBackend
using LinearAlgebra

# Module-level backend for KA kernel calls
const _GPU_BACKEND = oneAPIBackend()

# Ceiling integer division (oneAPI JIT-safe chunking helper)
cdiv(a::Int, b::Int) = (a + b - 1) ÷ b

export fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!
export fused_ssm_gate_sigmoid!, fused_ssm_decay!, gpu_argmax!, gpu_sample!
export reduce_sum_kernel!, reduce_sum!, reduce_sum_sq!, scale_kernel!, scale!, elementwise_mul_kernel!, elementwise_mul!

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
    # Use our kernel instead of BLAS dot — computes sum-of-squares in one kernel
    ss = reduce_sum_sq!(x)
    inv_norm = one(T) / sqrt(ss + eps)
    kfn = l2norm_scale_kernel!(_GPU_BACKEND)
    kfn(x, inv_norm; ndrange=(n,))
    return x
end

# ============================================================
# Reduction Kernel (GPU-safe sum — no BLAS dependency)
# Chunked for large arrays to stay below the oneAPI JIT loop bound.
# ============================================================
@kernel function reduce_sum_kernel!(out, x, n_chunks::Int, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        start = (c - 1) * chunk_size + 1
        stop = min(c * chunk_size, length(x))
        s = zero(eltype(x))
        for j in start:stop  # <= 32 iterations
            s += x[j]
        end
        @inbounds out[c] = s
    end
end

function reduce_sum!(x::AbstractArray{T,1}, chunk_size::Int=32) where T
    n = length(x)
    if n <= 60
        out = oneAPI.oneArray{T}(undef, 1)
        kfn = reduce_sum_kernel!(_GPU_BACKEND)
        kfn(out, x, 1, chunk_size; ndrange=(1,))
        oneAPI.oneL0.synchronize()
        return Array(out)[1]
    end
    n_chunks = cdiv(n, chunk_size)
    out = oneAPI.oneArray(zeros(T, n_chunks))
    kfn = reduce_sum_kernel!(_GPU_BACKEND)
    kfn(out, x, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    return sum(Array(out))
end

function reduce_sum!(x::AbstractArray{T}) where T
    return reduce_sum!(vec(x))
end

# ============================================================
# Reduction Kernel (sum of squares — avoids broken broadcast)
# Chunked for large arrays to stay below the oneAPI JIT loop bound.
# ============================================================
@kernel function reduce_sum_sq_kernel!(out, x, n_chunks::Int, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        start = (c - 1) * chunk_size + 1
        stop = min(c * chunk_size, length(x))
        s = zero(eltype(x))
        for j in start:stop  # <= 32 iterations
            s += x[j] * x[j]
        end
        @inbounds out[c] = s
    end
end

function reduce_sum_sq!(x::AbstractArray{T,1}, chunk_size::Int=32) where T
    n = length(x)
    if n <= 60
        out = oneAPI.oneArray{T}(undef, 1)
        kfn = reduce_sum_sq_kernel!(_GPU_BACKEND)
        kfn(out, x, 1, chunk_size; ndrange=(1,))
        oneAPI.oneL0.synchronize()
        return Array(out)[1]
    end
    n_chunks = cdiv(n, chunk_size)
    out = oneAPI.oneArray(zeros(T, n_chunks))
    kfn = reduce_sum_sq_kernel!(_GPU_BACKEND)
    kfn(out, x, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    return sum(Array(out))
end

function reduce_sum_sq!(x::AbstractArray{T}) where T
    return reduce_sum_sq!(vec(x))
end

# ============================================================
# Scale Kernel (x[i] = y[i] * s * w[i])
# No broadcast — uses KA kernel
# ============================================================
@kernel function scale_kernel!(out, x, scale, weight)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = x[i] * scale * weight[i]
    end
end

function scale!(out, x, scale, weight)
    n = length(out)
    kfn = scale_kernel!(_GPU_BACKEND)
    kfn(out, x, scale, weight; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Element-wise Multiply Kernel (out[i] = a[i] * b[i])
# ============================================================
@kernel function elementwise_mul_kernel!(out, a, b)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = a[i] * b[i]
    end
end

function elementwise_mul!(out, a, b)
    n = length(out)
    kfn = elementwise_mul_kernel!(_GPU_BACKEND)
    kfn(out, a, b; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out
end

# ============================================================
# Attention weighted sum across all heads (type generic)
# Chunked over seq_len so no thread exceeds the oneAPI JIT bound.
# Replaces per-head loop that allocates weighted_v temporary.
# ============================================================
@kernel function attention_weighted_sum_kernel!(attn_out_partial, scores, v_cache, n_heads::Int, head_dim::Int,
                                                seq_len::Int, n_groups::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * head_dim
    if idx <= n_total * n_chunks
        h = (idx - 1) ÷ (head_dim * n_chunks)
        d = ((idx - 1) % (head_dim * n_chunks)) ÷ n_chunks
        c = (idx - 1) % n_chunks
        T = eltype(attn_out_partial)
        if h < n_heads && d < head_dim
            kv_h = h ÷ n_groups
            k_off = kv_h * head_dim + d
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, seq_len)
            acc = zero(T)
            for s in chunk_start:chunk_end  # <= 32 iterations
                acc += scores[h + 1, s] * v_cache[k_off + 1, s]
            end
            @inbounds attn_out_partial[h + 1, d + 1, c + 1] = acc
        end
    end
end

function fused_attention_weighted_sum!(
    attn_out::AbstractArray{T,1},
    scores::AbstractArray{T,2},
    v_cache::AbstractArray{T,2},
    n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int,
    chunk_size::Int=32
) where T
    n_total = n_heads * head_dim
    n_total == 0 && return attn_out
    n_chunks = cdiv(seq_len, chunk_size)
    attn_out_partial = oneAPI.oneArray(zeros(T, n_heads, head_dim, n_chunks))
    kfn = attention_weighted_sum_kernel!(_GPU_BACKEND)
    kfn(attn_out_partial, scores, v_cache, n_heads, head_dim, seq_len, n_groups, n_chunks;
        ndrange=(n_total * n_chunks,))
    oneAPI.oneL0.synchronize()
    copyto!(attn_out, oneAPI.oneArray(sum(Array(attn_out_partial), dims=3)[:]))
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
# GPU-native argmax (chunked — single-thread O(n) hits the oneAPI
# JIT loop bound on vocab-size logits (~151k)).
# Pass 1: per-chunk (max, index); host picks the global result.
# ============================================================
@kernel function argmax_chunk_kernel!(chunk_max, chunk_idx, logits, n_chunks::Int, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        start = (c - 1) * chunk_size + 1
        stop = min(c * chunk_size, length(logits))
        T = eltype(logits)
        best_i = start
        best_val = logits[start]
        for i in start+1:stop  # <= 32 iterations
            v = logits[i]
            if v > best_val
                best_val = v
                best_i = i
            end
        end
        @inbounds chunk_max[c] = best_val
        @inbounds chunk_idx[c] = best_i
    end
end

function gpu_argmax!(logits::AbstractArray{T,1}, chunk_size::Int=32) where T
    n = length(logits)
    n == 0 && return 1
    n_chunks = cdiv(n, chunk_size)
    chunk_max = oneAPI.oneArray(zeros(T, n_chunks))
    chunk_idx = oneAPI.oneArray(zeros(Int, n_chunks))
    kfn = argmax_chunk_kernel!(_GPU_BACKEND)
    kfn(chunk_max, chunk_idx, logits, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    cmax = Array(chunk_max); cidx = Array(chunk_idx)
    best_i = cidx[1]
    best_val = cmax[1]
    for c in 2:n_chunks
        if cmax[c] > best_val
            best_val = cmax[c]
            best_i = cidx[c]
        end
    end
    return Int(best_i)
end

# ============================================================
# GPU-native temperature + softmax (single thread for stability)
# Computes probability vector on GPU without CPU sync.
# Chunked over the vocab dimension to stay below the JIT bound.
# ============================================================
@kernel function temp_softmax_pass1_max!(max_partial, logits, n, n_chunks, chunk_size::Int, temperature::T) where T
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        chunk_start = (c - 1) * chunk_size + 1
        chunk_end = min(c * chunk_size, n)
        m = typemin(T)
        for i in chunk_start:chunk_end  # <= 32 iterations
            v = logits[i] / temperature
            if v > m
                m = v
            end
        end
        @inbounds max_partial[c] = m
    end
end

@kernel function temp_softmax_pass2!(out_probs, logits, temperature, max_val)
    i = @index(Global, Linear)
    T = eltype(logits)
    @inbounds out_probs[i] = exp(logits[i] / temperature - T(max_val))
end

@kernel function temp_softmax_pass3_sum!(sum_partial, out_probs, n, n_chunks, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        chunk_start = (c - 1) * chunk_size + 1
        chunk_end = min(c * chunk_size, n)
        T = eltype(out_probs)
        ss = zero(T)
        for i in chunk_start:chunk_end  # <= 32 iterations
            ss += out_probs[i]
        end
        @inbounds sum_partial[c] = ss
    end
end

@kernel function temp_softmax_pass4!(out_probs, out_probs_norm, sum_val)
    i = @index(Global, Linear)
    T = eltype(out_probs)
    @inbounds out_probs_norm[i] = out_probs[i] / T(sum_val)
end

function temperature_softmax_kernel!(logits, temperature, out_probs)
    n = length(logits)
    n == 0 && return out_probs
    T = eltype(logits)
    if n <= 60
        # small vocab: single-threaded path is safe below the JIT threshold
        if n == 1
            out_probs[1] = one(T)
            return out_probs
        end
        kfn = temp_softmax_kernel!_single_threaded(_GPU_BACKEND)
        kfn(out_probs, logits, temperature; ndrange=(1,))
        oneAPI.oneL0.synchronize()
        return out_probs
    end
    n_chunks = cdiv(n, chunk_size)
    # Pass 1: max over logits/temperature chunks (host picks global max)
    max_partial = oneAPI.oneArray(zeros(T, n_chunks))
    kfn = temp_softmax_pass1_max!(_GPU_BACKEND)
    kfn(max_partial, logits, n, n_chunks, chunk_size, temperature; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    max_val = maximum(Array(max_partial))
    # Pass 2: out_probs = exp(logits/temp - max)
    kfn = temp_softmax_pass2!(_GPU_BACKEND)
    kfn(out_probs, logits, temperature, max_val; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    # Pass 3: sum over chunks (host computes total)
    sum_partial = oneAPI.oneArray(zeros(T, n_chunks))
    kfn = temp_softmax_pass3_sum!(_GPU_BACKEND)
    kfn(sum_partial, out_probs, n, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    sum_val = sum(Array(sum_partial))
    # Pass 4: normalize
    kfn = temp_softmax_pass4!(_GPU_BACKEND)
    kfn(out_probs, out_probs, sum_val; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return out_probs
end

# Single-threaded temperature softmax (kept for small vocab fallback)
@kernel function temp_softmax_kernel!_single_threaded(out_probs, logits, temperature)
    n = length(logits)
    # Single-threaded for numerical stability
    @inbounds for i in 1:n
        out_probs[i] = logits[i] / temperature
    end
    max_val = out_probs[1]
    @inbounds for i in 2:n
        if out_probs[i] > max_val
            max_val = out_probs[i]
        end
    end
    sum_exp = zero(eltype(logits))
    @inbounds for i in 1:n
        exp_val = exp(out_probs[i] - max_val)
        out_probs[i] = exp_val
        sum_exp += exp_val
    end
    inv_sum = one(eltype(logits)) / sum_exp
    @inbounds for i in 1:n
        out_probs[i] *= inv_sum
    end
end

# ============================================================
# GPU-native search over cumulative probabilities
# Single thread, O(n), returns index of first element where cumsum >= r.
# Chunked: one GPU pass computes per-chunk sums; host locates the
# crossing chunk and scans only that chunk's <=32 elements.
# ============================================================
@kernel function search_cum_kernel!(chunk_sum, probs, n_chunks::Int, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        start = (c - 1) * chunk_size + 1
        stop = min(c * chunk_size, length(probs))
        s = zero(eltype(probs))
        for i in start:stop  # <= 32 iterations
            s += probs[i]
        end
        @inbounds chunk_sum[c] = s
    end
end

function gpu_search_kernel!(probs, r, result)
    n = length(probs)
    n == 0 && (result[1] = n; return result)
    n_chunks = cdiv(n, 32)
    chunk_sum = oneAPI.oneArray(zeros(eltype(probs), n_chunks))
    kfn = search_cum_kernel!(_GPU_BACKEND)
    kfn(chunk_sum, probs, n_chunks, 32; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    cs = Array(chunk_sum)
    # Host: locate the chunk containing the crossing point
    cum = zero(eltype(probs))
    hit = nothing
    for c in 1:n_chunks
        prev = cum
        cum += cs[c]
        if cum >= r && hit === nothing
            # scan within this chunk (<= 32 elements)
            start = (c - 1) * 32 + 1
            stop = min(c * 32, n)
            acc = prev
            for i in start:stop
                acc += probs[i]
                if r <= acc
                    hit = i
                    break
                end
            end
            break
        end
    end
    result[1] = hit === nothing ? n : Int(hit)
    return result
end

# ============================================================
# GPU-native sampling — no CPU-GPU sync of probability vector!
# Only syncs a single Int back to CPU.
# ============================================================
function gpu_sample!(logits::AbstractArray{T,1}, temperature::S) where {T <: AbstractFloat, S <: AbstractFloat}
    n = length(logits)
    out_probs = oneAPI.oneArray{T}(undef, n)
    
    temp = convert(T, temperature)
    
    # GPU: temperature scaling + softmax (chunked, single thread equivalent)
    softmax_kfn = temperature_softmax_kernel!(_GPU_BACKEND)
    softmax_kfn(logits, temp, out_probs)
    
    # CPU: random number (only 4 bytes, negligible)
    r = rand(T)
    
    # GPU: binary search for sampled index
    result = oneAPI.oneArray{Int}(undef)
    search_kfn = gpu_search_kernel!(_GPU_BACKEND)
    search_kfn(out_probs, r, result)
    
    # GPU->CPU: single Int (O(1) transfer)
    return oneAPI.Array(result)[1]
end

end # module
