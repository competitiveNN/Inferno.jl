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

# Ceiling integer division: rounds up to the nearest whole chunk.
# Used to split large per-head/row work into chunks of at most `chunk_size`,
# keeping every compile-time kernel loop bound < ~57 (oneAPI JIT bug at >= 64).
cdiv(a::Int, b::Int) = (a + b - 1) ÷ b

export rmsnorm_kernel!, gelu_kernel!, residual_add_kernel!, apply_rope_kernel!
export softmax_kernel!, matmul_vec_kernel!, silu_kernel!, sigmoid_kernel!
export batched_attention_scores_kernel!, batched_softmax_kernel!, batched_ssm_state_kernel!
export rmsnorm_gpu!, gelu_gpu!, residual_add_gpu!, apply_rope_gpu!, softmax_gpu!, silu_gpu!, sigmoid_gpu!
export batched_attention_scores!, batched_softmax!, fused_attention_forward!
export ssm_conv_kernel!, ssm_qk_norm_kernel!, ssm_state_kernel!, ssm_y_norm_kernel!
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
# Per-head RMSNorm (Qwen3.5 attention Q/K normalization)
# Each head's head_dim elements are normalized separately with the
# same head_dim-length weight vector (matches CPU rmsnorm_rotary).
# ============================================================
# Pass 1: partial sum-of-squares per chunk per head (compile-time loop == chunk_size).
# Host reduces partials into a per-head inv_rms.
@kernel function rmsnorm_headed_pass1!(partial, x, head_dim, n_heads, n_chunks, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_heads * n_chunks
        T_acc = Float32
        h = (idx - 1) ÷ n_chunks
        c = (idx - 1) % n_chunks
        base = h * head_dim + c * chunk_size + 1
        head_end = (h + 1) * head_dim
        ss = zero(T_acc)
        for k in 0:(chunk_size - 1)  # <= 32 iterations
            i = base + k
            if i <= head_end
                ss += Float32(x[i])^2
            end
        end
        @inbounds partial[h + 1, c + 1] = ss
    end
end

# Pass 2: apply per-head normalization to each chunk.
@kernel function rmsnorm_headed_pass2!(out, x, weight, inv_rms_head, head_dim, n_heads, n_chunks, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_heads * n_chunks
        T = eltype(x)
        h = (idx - 1) ÷ n_chunks
        c = (idx - 1) % n_chunks
        base = h * head_dim + c * chunk_size + 1
        head_end = (h + 1) * head_dim
        inv = T(inv_rms_head[h + 1])
        for k in 0:(chunk_size - 1)  # <= 32 iterations
            i = base + k
            if i <= head_end
                d = (i - 1) % head_dim + 1
                out[i] = x[i] * inv * weight[d]
            end
        end
    end
end

function rmsnorm_headed_gpu!(out::AbstractArray, x::AbstractArray, weight::AbstractArray,
                             eps::AbstractFloat, head_dim::Int, chunk_size::Int=32)
    n = length(x)
    n == 0 && return out
    n_heads = n ÷ head_dim
    n_chunks = cdiv(head_dim, chunk_size)
    T_x = eltype(x)
    T_acc = promote_type(T_x, Float32)  # accumulate in FP32 for precision
    partial = oneAPI.oneArray(zeros(T_acc, n_heads, n_chunks))
    kfn = rmsnorm_headed_pass1!(_GPU_BACKEND)
    kfn(partial, x, head_dim, n_heads, n_chunks, chunk_size; ndrange=(n_heads * n_chunks,))
    oneAPI.oneL0.synchronize()
    ps = Array(partial)  # (n_heads, n_chunks) back on host
    ss_head = sum(ps, dims=2)[:]  # (n_heads,)
    inv_rms_head = oneAPI.oneArray(Float32(1) ./ sqrt.(ss_head ./ head_dim .+ eps))
    kfn2 = rmsnorm_headed_pass2!(_GPU_BACKEND)
    kfn2(out, x, weight, inv_rms_head, head_dim, n_heads, n_chunks, chunk_size; ndrange=(n_heads * n_chunks,))
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
# Stable softmax over a single row. For n <= 60 a single
# thread is safe; larger rows are chunked to avoid the
# oneAPI JIT bug on per-thread loops >= ~57.
# ============================================================
@kernel function softmax_pass1_max!(max_partial, x, n, n_chunks, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        chunk_start = (c - 1) * chunk_size + 1
        chunk_end = min(c * chunk_size, n)
        T = eltype(x)
        m = typemin(T)
        for j in chunk_start:chunk_end  # <= 32 iterations
            v = x[j]
            if v > m
                m = v
            end
        end
        @inbounds max_partial[c] = m
    end
end

@kernel function softmax_pass2_exp!(out, x, max_val)
    i = @index(Global, Linear)
    T = eltype(x)
    @inbounds out[i] = exp(x[i] - T(max_val))
end

@kernel function softmax_pass3_sum!(sum_partial, x, n, n_chunks, chunk_size::Int)
    idx = @index(Global, Linear)
    if idx <= n_chunks
        c = idx
        chunk_start = (c - 1) * chunk_size + 1
        chunk_end = min(c * chunk_size, n)
        T = eltype(x)
        ss = zero(T)
        for j in chunk_start:chunk_end  # <= 32 iterations
            ss += x[j]
        end
        @inbounds sum_partial[c] = ss
    end
end

@kernel function softmax_pass4_norm!(out, x, sum_val)
    i = @index(Global, Linear)
    T = eltype(x)
    @inbounds out[i] = x[i] / T(sum_val)
end

# Single-threaded stable softmax (original implementation): safe only for n <= ~57.
# Kept for backwards compatibility; softmax_gpu! falls back to it for small rows.
@kernel function softmax_kernel!_single_threaded(out, x)
    i = @index(Global, Linear)
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

function softmax_gpu!(out::AbstractArray, x::AbstractArray, chunk_size::Int=32)
    n = length(x)
    n == 0 && return out
    if n <= 60
        # single-threaded softmax is safe below the JIT loop threshold
        kfn = softmax_kernel!_single_threaded(_GPU_BACKEND)
        kfn(out, x; ndrange=(1,))
        oneAPI.oneL0.synchronize()
        return out
    end
    n_chunks = cdiv(n, chunk_size)
    T = eltype(x)
    # Pass 1: max over each chunk (host picks the global max)
    max_partial = oneAPI.oneArray(zeros(T, n_chunks))
    kfn = softmax_pass1_max!(_GPU_BACKEND)
    kfn(max_partial, x, n, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    max_val = maximum(Array(max_partial))
    # Pass 2: exp(x - max)
    kfn2 = softmax_pass2_exp!(_GPU_BACKEND)
    kfn2(out, x, max_val; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    # Pass 3: sum of exp over chunks (host computes the total)
    sum_partial = oneAPI.oneArray(zeros(T, n_chunks))
    kfn3 = softmax_pass3_sum!(_GPU_BACKEND)
    kfn3(sum_partial, out, n, n_chunks, chunk_size; ndrange=(n_chunks,))
    oneAPI.oneL0.synchronize()
    sum_val = sum(Array(sum_partial))
    # Pass 4: normalize
    kfn4 = softmax_pass4_norm!(_GPU_BACKEND)
    kfn4(out, out, sum_val; ndrange=(n,))
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
# Computes dot(Q[h], K_cache[kv_h, :, s]) for all heads in parallel,
# chunked over head_dim so no thread exceeds the oneAPI JIT bound.
# ============================================================
# Pass 1: partial dot products over head_dim chunks.
@kernel function batched_attention_scores_pass1!(scores_partial, q, k_cache, n_heads::Int, head_dim::Int,
                                                  seq_len::Int, n_groups::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * seq_len
    if idx <= n_total * n_chunks
        h = (idx - 1) ÷ (seq_len * n_chunks)
        s = ((idx - 1) % (seq_len * n_chunks)) ÷ n_chunks
        c = (idx - 1) % n_chunks
        T = eltype(scores_partial)
        if h < n_heads && s < seq_len
            kv_h = h ÷ n_groups
            q_off = h * head_dim
            k_off = kv_h * head_dim
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, head_dim)
            dot_sum = zero(T)
            for j in chunk_start:chunk_end  # <= 32 FMA
                dot_sum += q[q_off + j] * k_cache[k_off + j, s + 1]
            end
            @inbounds scores_partial[h + 1, s + 1, c + 1] = dot_sum
        end
    end
end

function batched_attention_scores!(scores::AbstractArray{T,2}, q::AbstractArray{T,1},
                                   k_cache::AbstractArray{T,2}, n_heads::Int, head_dim::Int,
                                   seq_len::Int, n_groups::Int, chunk_size::Int=32) where T
    n_total = n_heads * seq_len
    n_chunks = cdiv(head_dim, chunk_size)
    scores_partial = oneAPI.oneArray(zeros(T, n_heads, seq_len, n_chunks))
    kfn = batched_attention_scores_pass1!(_GPU_BACKEND)
    kfn(scores_partial, q, k_cache, n_heads, head_dim, seq_len, n_groups, n_chunks;
        ndrange=(n_total * n_chunks,))
    oneAPI.oneL0.synchronize()
    host_scores = reshape(sum(Array(scores_partial), dims=3)[:] / sqrt(T(head_dim)), size(scores))
    if scores isa oneAPI.oneArray
        copyto!(scores, oneAPI.oneArray(host_scores))
    else
        copyto!(scores, host_scores)
    end
    return scores
end

# ============================================================
# Batched Softmax Kernel (type generic)
# Normalizes each head's row over seq_len positions using the
# stable max-sum trick, chunked so no thread exceeds the JIT bound.
# ============================================================
@kernel function batched_softmax_pass1_max!(row_max_partial, scores, n_heads::Int, seq_len::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * seq_len
    if idx <= n_total * n_chunks
        h = (idx - 1) ÷ (seq_len * n_chunks)
        s = ((idx - 1) % (seq_len * n_chunks)) ÷ n_chunks
        c = (idx - 1) % n_chunks
        T = eltype(scores)
        if h < n_heads && s < seq_len
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, seq_len)
            m = typemin(T)
            for j in chunk_start:chunk_end  # <= 32 iterations
                v = scores[h + 1, j]
                if v > m
                    m = v
                end
            end
            @inbounds row_max_partial[h + 1, s + 1, c + 1] = m
        end
    end
end

@kernel function batched_softmax_pass2_exp!(scores, row_max, n_heads::Int, seq_len::Int)
    idx = @index(Global, Linear)
    if idx <= n_heads * seq_len
        h = (idx - 1) ÷ seq_len
        s = (idx - 1) % seq_len
        scores[idx] = exp(scores[idx] - row_max[h + 1])
    end
end

@kernel function batched_softmax_pass3_sum!(row_sum_partial, scores, n_heads::Int, seq_len::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * seq_len
    if idx <= n_total * n_chunks
        h = (idx - 1) ÷ (seq_len * n_chunks)
        s = ((idx - 1) % (seq_len * n_chunks)) ÷ n_chunks
        c = (idx - 1) % n_chunks
        T = eltype(scores)
        if h < n_heads && s < seq_len
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, seq_len)
            ss = zero(T)
            for j in chunk_start:chunk_end  # <= 32 iterations
                ss += scores[h + 1, j]
            end
            @inbounds row_sum_partial[h + 1, s + 1, c + 1] = ss
        end
    end
end

@kernel function batched_softmax_pass4_norm!(scores, row_sum, n_heads::Int, seq_len::Int)
    idx = @index(Global, Linear)
    if idx <= n_heads * seq_len
        h = (idx - 1) ÷ seq_len
        s = (idx - 1) % seq_len
        scores[idx] = scores[idx] / row_sum[h + 1]
    end
end

function batched_softmax!(scores::AbstractArray{T,2}, n_heads::Int, seq_len::Int, chunk_size::Int=32) where T
    n_total = n_heads * seq_len
    if n_total == 0
        return scores
    end
    n_chunks = cdiv(seq_len, chunk_size)
    if n_chunks == 1
        # small seq_len: single-threaded softmax is safe below the JIT threshold
        kfn = softmax_kernel!_single_threaded(_GPU_BACKEND)
        kfn(scores, scores; ndrange=(1,))
        oneAPI.oneL0.synchronize()
        return scores
    end
    # Pass 1: max over each seq_len chunk -> host picks global row max
    row_max_partial = oneAPI.oneArray(zeros(T, n_heads, seq_len, n_chunks))
    kfn = batched_softmax_pass1_max!(_GPU_BACKEND)
    kfn(row_max_partial, scores, n_heads, seq_len, n_chunks; ndrange=(n_total * n_chunks,))
    oneAPI.oneL0.synchronize()
    row_max = maximum(Array(row_max_partial), dims=3)[:]  # (n_heads,)
    # Pass 2: exp(scores - row_max)
    kfn2 = batched_softmax_pass2_exp!(_GPU_BACKEND)
    kfn2(scores, row_max, n_heads, seq_len; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    # Pass 3: sum over chunks -> host computes row sum
    row_sum_partial = oneAPI.oneArray(zeros(T, n_heads, seq_len, n_chunks))
    kfn3 = batched_softmax_pass3_sum!(_GPU_BACKEND)
    kfn3(row_sum_partial, scores, n_heads, seq_len, n_chunks; ndrange=(n_total * n_chunks,))
    oneAPI.oneL0.synchronize()
    row_sum = sum(Array(row_sum_partial), dims=3)[:]  # (n_heads,)
    # Pass 4: normalize
    kfn4 = batched_softmax_pass4_norm!(_GPU_BACKEND)
    kfn4(scores, row_sum, n_heads, seq_len; ndrange=(n_total,))
    oneAPI.oneL0.synchronize()
    return scores
end
# ============================================================
# SSM Conv1D kernel (Qwen3.5 GatedDeltaNet): ring-buffer shift,
# x_conv[c] = silu(dot(shifted_state[c], conv1d[c])), and store the
# new in_proj output at the last column. Matches CPU exactly.
# The raw in_proj output is read at conv_state[c,K] BEFORE xz[c] is
# overwritten with x_conv[c], so a single buffer suffices.
# ============================================================
@kernel function ssm_conv_kernel!(conv_state, conv1d, xz,
                                  conv_channels::Int, conv_kernel::Int)
    c = @index(Global, Linear)
    if c <= conv_channels
        T = eltype(xz)
        # Shift the ring buffer left by one step
        for k in 1:(conv_kernel - 1)
            conv_state[c, k] = conv_state[c, k + 1]
        end
        # Store the new (raw) in_proj output at the last column (Float16 -> Float32).
        conv_state[c, conv_kernel] = xz[c]
        # Convolution over the shifted state: compute in Float32 for accuracy
        # (conv_state/conv1d are Float32), then fuse SiLU and cast back to xz.
        v = zero(Float32)
        for k in 1:(conv_kernel - 1)
            v += conv_state[c, k] * conv1d[c, k]
        end
        v += conv_state[c, conv_kernel] * conv1d[c, conv_kernel]
        # Fused conv + SiLU in Float32, stored back to xz as Float16
        xz[c] = T(v / (one(Float32) + exp(-v)))
    end
end

function ssm_conv_kernel!(conv_state::AbstractArray{T,2}, conv1d::AbstractArray{T,2},
                          xz::AbstractArray{<:AbstractFloat,1},
                          conv_channels::Int, conv_kernel::Int) where T
    kfn = ssm_conv_kernel!(_GPU_BACKEND)
    kfn(conv_state, conv1d, xz, conv_channels, conv_kernel; ndrange=(conv_channels,))
    oneAPI.oneL0.synchronize()
    return xz
end

# ============================================================
# SSM per-head L2 normalization (Qwen3.5 GatedDeltaNet).
# For each v_head h with group g = ((h-1) % num_k_heads) + 1:
#   qg = xz[g*head_k_dim + i], kg = xz[qk_size + g*head_k_dim + i]
#   q_norm = qg / (sqrt(sum_q) + eps) * scale,
#   k_norm = kg / (sqrt(sum_k) + eps)
# Matches the CPU delta-net L2 normalization (scale = 1/sqrt(head_k_dim)
# applies only to q, eps is added after sqrt, as in llama.cpp ggml_l2_norm).
# Chunked over head_k_dim so no thread exceeds the oneAPI JIT bound.
# ============================================================
# Pass 1: partial sum-of-squares for q and k per chunk per v_head.
@kernel function ssm_qk_norm_pass1!(partial_q, partial_k, xz, qk_size::Int, head_k_dim::Int,
                                    num_k_heads::Int, num_v_heads::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * n_chunks
        h = (idx - 1) ÷ n_chunks          # 0-based v-head
        c = (idx - 1) % n_chunks
        g = h % num_k_heads               # 0-based k-group (CPU: ((h+1)-1)%num_k_heads)
        qg_base = g * head_k_dim
        T_x = eltype(xz)
        T_n = promote_type(T_x, Float32)  # accumulate in FP32 (CPU is Float32)
        chunk_start = c * 32 + 1
        chunk_end = min((c + 1) * 32, head_k_dim)
        ss_q = zero(T_n); ss_k = zero(T_n)
        for j in chunk_start:chunk_end  # <= 32 iterations
            qg = T_n(xz[qg_base + j])
            kg = T_n(xz[qk_size + qg_base + j])
            ss_q += qg * qg
            ss_k += kg * kg
        end
        @inbounds partial_q[h + 1, c + 1] = ss_q
        @inbounds partial_k[h + 1, c + 1] = ss_k
    end
end

# Pass 2: apply per-head normalization to each chunk.
@kernel function ssm_qk_norm_pass2!(q_norm, k_norm, xz, qk_size::Int, head_k_dim::Int,
                                    num_v_heads::Int, num_k_heads::Int,
                                    qn::AbstractArray{Float32,1}, ki::AbstractArray{Float32,1}, n_chunks::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * n_chunks
        T_x = eltype(xz)
        h = (idx - 1) ÷ n_chunks          # 0-based v-head (write target)
        c = (idx - 1) % n_chunks
        g = h % num_k_heads               # 0-based k-group (read source)
        out_base = h * head_k_dim
        qg_base = g * head_k_dim
        qn_h = qn[h + 1]
        ki_h = ki[h + 1]
        chunk_start = c * 32 + 1
        chunk_end = min((c + 1) * 32, head_k_dim)
        for j in chunk_start:chunk_end  # <= 32 iterations
            qg = xz[qg_base + j]
            kg = xz[qk_size + qg_base + j]
            q_norm[out_base + j] = qg * qn_h
            k_norm[out_base + j] = kg * ki_h
        end
    end
end

function ssm_qk_norm_kernel!(xz::AbstractArray{<:AbstractFloat,1}, qk_size::Int, head_k_dim::Int,
                             num_k_heads::Int, head_v_dim::Int, num_v_heads::Int,
                             eps::AbstractFloat, scale::AbstractFloat,
                             q_norm::AbstractArray{<:AbstractFloat,1}, k_norm::AbstractArray{<:AbstractFloat,1},
                             chunk_size::Int=32)
    n_chunks = cdiv(head_k_dim, chunk_size)
    T_x = eltype(xz)
    T_acc = promote_type(T_x, Float32)
    partial_q = oneAPI.oneArray(zeros(T_acc, num_v_heads, n_chunks))
    partial_k = oneAPI.oneArray(zeros(T_acc, num_v_heads, n_chunks))
    kfn = ssm_qk_norm_pass1!(_GPU_BACKEND)
    kfn(partial_q, partial_k, xz, qk_size, head_k_dim, num_k_heads, num_v_heads, n_chunks;
        ndrange=(num_v_heads * n_chunks,))
    oneAPI.oneL0.synchronize()
    pq = Array(partial_q); pk = Array(partial_k)
    sq = sum(pq, dims=2)[:]
    sk = sum(pk, dims=2)[:]
    qn = scale ./ (sqrt.(sq) .+ eps)
    ki = one(T_x) ./ (sqrt.(sk) .+ eps)
    qn_dev = oneAPI.oneArray(Vector{T_acc}(qn))
    ki_dev = oneAPI.oneArray(Vector{T_acc}(ki))
    kfn2 = ssm_qk_norm_pass2!(_GPU_BACKEND)
    kfn2(q_norm, k_norm, xz, qk_size, head_k_dim, num_v_heads, num_k_heads, qn_dev, ki_dev, n_chunks;
        ndrange=(num_v_heads * n_chunks,))
    oneAPI.oneL0.synchronize()
    return (q_norm, k_norm)
end

# ============================================================
# SSM state-update kernel (Qwen3.5 GatedDeltaNet delta-net).
# For each (v_head h, v-dim i):
#   d  = beta_gate[h] * (v - sk), where sk = state * k_norm (old state)
#   state[i,:,h] = state[i,:,h] * decay[h] + d * k_norm[h]'  (rank-1 ger update)
#   y  = state[i, :, h] * q_norm[h]   (state after the ger update)
#
# Chunked over head_k_dim so no thread exceeds the oneAPI JIT bound:
#   Pass 1: sk = state * k_norm  (partial sums over head_k_dim chunks -> host reduce)
#   Pass 2: d = beta_gate * (v - sk)
#   Pass 3: in-place ger (state *= decay + d * k_norm') + partial y sums -> host reduce
#   Pass 4: copy y -> y_out
# ============================================================
@kernel function ssm_state_pass1_sk!(sk_partial, h_state, k_norm, num_v_heads::Int, head_v_dim::Int,
                                     head_k_dim::Int, num_k_heads::Int, n_chunks::Int, decay::AbstractArray{<:AbstractFloat,1})
    idx = @index(Global, Linear)
    n_total = num_v_heads * head_v_dim * n_chunks
    if idx <= n_total
        h = (idx - 1) ÷ (head_v_dim * n_chunks)
        rem = (idx - 1) % (head_v_dim * n_chunks)
        i = rem ÷ n_chunks
        c = rem % n_chunks
        k_base = h * head_k_dim
        chunk_start = c * 32 + 1
        chunk_end = min((c + 1) * 32, head_k_dim)
        T = eltype(h_state)
        sk = zero(T)
        for j in chunk_start:chunk_end  # <= 32 iterations
            # CPU applies state *= decay BEFORE sk = state * k; fold the per-head
            # scalar decay into the accumulation (same rounding, avoids extra pass).
            sk += decay[h + 1] * h_state[i + 1, j, h + 1] * k_norm[k_base + j]
        end
        @inbounds sk_partial[h + 1, i + 1, c + 1] = sk
    end
end

@kernel function ssm_state_pass2_d!(d, beta_gate, sk, xz, v_offset, num_v_heads::Int, head_v_dim::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * head_v_dim
        h = (idx - 1) ÷ head_v_dim
        i = (idx - 1) % head_v_dim
        # v lives at 2*qk_size + h*head_v_dim + i (GatedDeltaNet: v_all =
        # reshape(xz[2*qk+1 : 2*qk+d_inner], head_v_dim, num_v_heads)) — matches CPU.
        d[h * head_v_dim + i + 1] = beta_gate[h + 1] * (xz[v_offset + h * head_v_dim + i + 1] - sk[h + 1, i + 1])
    end
end

@kernel function ssm_state_pass3!(h_state, decay, d, k_norm, q_norm, y_partial, num_v_heads::Int,
                                  head_v_dim::Int, head_k_dim::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = num_v_heads * head_v_dim * n_chunks
    if idx <= n_total
        h = (idx - 1) ÷ (head_v_dim * n_chunks)
        rem = (idx - 1) % (head_v_dim * n_chunks)
        i = rem ÷ n_chunks
        c = rem % n_chunks
        decay_val = decay[h + 1]
        d_val = d[h * head_v_dim + i + 1]
        k_base = h * head_k_dim
        chunk_start = c * 32 + 1
        chunk_end = min((c + 1) * 32, head_k_dim)
        T = eltype(h_state)
        acc = zero(T)
        for j in chunk_start:chunk_end  # <= 32 iterations
            s_new = h_state[i + 1, j, h + 1] * decay_val + d_val * k_norm[k_base + j]
            h_state[i + 1, j, h + 1] = s_new        # in-place ger update (state persists)
            acc += s_new * q_norm[k_base + j]
        end
        @inbounds y_partial[h + 1, i + 1, c + 1] = acc
    end
end

@kernel function ssm_state_pass4!(y_out, y, num_v_heads::Int, head_v_dim::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * head_v_dim
        h = (idx - 1) ÷ head_v_dim
        i = (idx - 1) % head_v_dim
        y_out[h * head_v_dim + i + 1] = y[h + 1, i + 1]
    end
end

function ssm_state_kernel!(h_state::AbstractArray{T,3},
                           decay::AbstractArray{<:AbstractFloat,1},
                           beta_gate::AbstractArray{<:AbstractFloat,1},
                           q_norm::AbstractArray{<:AbstractFloat,1},
                           k_norm::AbstractArray{<:AbstractFloat,1},
                           xz::AbstractArray{<:AbstractFloat,1},
                           qk_size::Int, num_k_heads::Int, head_k_dim::Int,
                           head_v_dim::Int, num_v_heads::Int,
                           y_out::AbstractArray{T,1}, chunk_size::Int=32) where T
    n_chunks = cdiv(head_k_dim, chunk_size)
    # Pass 1: sk = state * k_norm  (partial over head_k_dim chunks, decay folded in
    # so sk matches CPU semantics: CPU decays state first, then sk = state * k).
    sk_partial = oneAPI.oneArray(zeros(T, num_v_heads, head_v_dim, n_chunks))
    kfn = ssm_state_pass1_sk!(_GPU_BACKEND)
    kfn(sk_partial, h_state, k_norm, num_v_heads, head_v_dim, head_k_dim, num_k_heads, n_chunks, decay;
        ndrange=(num_v_heads * head_v_dim * n_chunks,))
    oneAPI.oneL0.synchronize()
    sk = reshape(sum(Array(sk_partial), dims=3)[:], num_v_heads, head_v_dim)
    sk = oneAPI.oneArray(sk)

    # Pass 2: d = beta_gate * (v - sk), v per head at 2*qk_size + h*head_v_dim + i
    d = oneAPI.oneArray(zeros(T, num_v_heads * head_v_dim))
    v_offset = 2 * qk_size
    kfn = ssm_state_pass2_d!(_GPU_BACKEND)
    kfn(d, beta_gate, sk, xz, v_offset, num_v_heads, head_v_dim; ndrange=(num_v_heads * head_v_dim,))
    oneAPI.oneL0.synchronize()

    # Pass 3: in-place ger (state *= decay + d * k_norm') + y partial sums
    y_partial = oneAPI.oneArray(zeros(T, num_v_heads, head_v_dim, n_chunks))
    kfn = ssm_state_pass3!(_GPU_BACKEND)
    kfn(h_state, decay, d, k_norm, q_norm, y_partial, num_v_heads, head_v_dim, head_k_dim, n_chunks;
        ndrange=(num_v_heads * head_v_dim * n_chunks,))
    oneAPI.oneL0.synchronize()
    y = reshape(sum(Array(y_partial), dims=3)[:], num_v_heads, head_v_dim)
    y = oneAPI.oneArray(y)

    # Pass 4: copy y -> y_out
    kfn = ssm_state_pass4!(_GPU_BACKEND)
    kfn(y_out, y, num_v_heads, head_v_dim; ndrange=(num_v_heads * head_v_dim,))
    oneAPI.oneL0.synchronize()
    return y_out
end

# ============================================================
# SSM per-head RMSNorm on the state output (Qwen3.5 GatedDeltaNet).
# Applies the same RMSNorm weight (head_v_dim,) to each head block,
# matching the CPU per-head rmsnorm_cpu! loop over num_v_heads.
# Chunked over head_v_dim so no thread exceeds the oneAPI JIT bound.
# ============================================================
# Pass 1: partial sum-of-squares per chunk per v_head.
@kernel function ssm_y_norm_pass1!(partial, y_out, head_v_dim::Int, num_v_heads::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * n_chunks
        h = (idx - 1) ÷ n_chunks
        c = (idx - 1) % n_chunks
        base = h * head_v_dim + c * 32 + 1
        head_end = (h + 1) * head_v_dim
        ss = Float32(0)
        for j in base:(base + 31)  # <= 32 iterations
            if j <= head_end
                ss += Float32(y_out[j]) ^ 2
            end
        end
        @inbounds partial[h + 1, c + 1] = ss
    end
end

# Pass 2: apply per-head normalization to each chunk.
@kernel function ssm_y_norm_pass2!(y_out, norm_w, inv, num_v_heads::Int, head_v_dim::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    if idx <= num_v_heads * n_chunks
        h = (idx - 1) ÷ n_chunks
        c = (idx - 1) % n_chunks
        base = h * head_v_dim + c * 32 + 1
        head_end = (h + 1) * head_v_dim
        inv_h = inv[h + 1]              # Float32 from host-computed inv
        for j in base:(base + 31)  # <= 32 iterations
            if j <= head_end
                d = (j - 1) % head_v_dim + 1
                y_out[j] = y_out[j] * inv_h * norm_w[d]
            end
        end
    end
end

function ssm_y_norm_kernel!(y_out::AbstractArray{T,1}, norm_w::AbstractArray{<:AbstractFloat,1},
                            eps::AbstractFloat, num_v_heads::Int, head_v_dim::Int, chunk_size::Int=32) where T
    n_chunks = cdiv(head_v_dim, chunk_size)
    T_acc = promote_type(T, Float32)
    partial = oneAPI.oneArray(zeros(T_acc, num_v_heads, n_chunks))
    kfn = ssm_y_norm_pass1!(_GPU_BACKEND)
    kfn(partial, y_out, head_v_dim, num_v_heads, n_chunks; ndrange=(num_v_heads * n_chunks,))
    oneAPI.oneL0.synchronize()
    ps = Array(partial)
    ss_head = sum(ps, dims=2)[:]
    inv = Float32(1) ./ sqrt.(ss_head ./ head_v_dim .+ eps)
    inv_dev = oneAPI.oneArray(inv)
    kfn2 = ssm_y_norm_pass2!(_GPU_BACKEND)
    kfn2(y_out, norm_w, inv_dev, num_v_heads, head_v_dim, n_chunks; ndrange=(num_v_heads * n_chunks,))
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
# Rewritten as chunked passes so no thread exceeds the oneAPI JIT bound:
#   Pass 1: scores[h,s] = dot(q_h, k_{kv_h,:,s}) / sqrt(head_dim),
#           chunked over head_dim -> host reduces into scores matrix
#   Host: stable softmax stats per head (max, sum_exp)
#   Pass 2: weighted_sum[h,d] = Σ_s prob[h,s] * v[kv_h,d,s],
#           chunked over seq_len -> host reduces into attn_out
# Type generic — works with Float16, Float32, etc.
# ============================================================
# Pass 1: partial dot products over head_dim chunks.
@kernel function fused_attention_scores_pass1!(scores_partial, q, k_cache, n_heads::Int, head_dim::Int,
                                               seq_len::Int, n_groups::Int, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * seq_len
    if idx <= n_total * n_chunks
        h = (idx - 1) ÷ (seq_len * n_chunks)
        s = ((idx - 1) % (seq_len * n_chunks)) ÷ n_chunks
        c = (idx - 1) % n_chunks
        T = eltype(scores_partial)
        if h < n_heads && s < seq_len
            kv_h = h ÷ n_groups
            q_off = h * head_dim
            k_off = kv_h * head_dim
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, head_dim)
            dot_sum = zero(T)
            for j in chunk_start:chunk_end  # <= 32 FMA
                dot_sum += q[q_off + j] * k_cache[k_off + j, s + 1]
            end
            @inbounds scores_partial[h + 1, s + 1, c + 1] = dot_sum
        end
    end
end

# Pass 2: weighted sum over seq_len chunks.
# Output layout is now (head_dim, n_heads, n_chunks): idx-1 = c + n_chunks*(d + head_dim*h).
@kernel function fused_attention_weighted_sum_pass!(attn_out_partial, scores, v_cache, n_heads::Int,
                                                    head_dim::Int, seq_len::Int, n_groups::Int,
                                                    row_max, row_sum, n_chunks::Int)
    idx = @index(Global, Linear)
    n_total = n_heads * head_dim
    if idx <= n_total * n_chunks
        c = (idx - 1) % n_chunks
        rem = (idx - 1) ÷ n_chunks
        d = rem % head_dim
        h = rem ÷ head_dim
        T = eltype(scores)
        if h < n_heads && d < head_dim
            kv_h = h ÷ n_groups
            v_off = kv_h * head_dim
            chunk_start = c * 32 + 1
            chunk_end = min((c + 1) * 32, seq_len)
            acc = zero(T)
            rm = T(row_max[h + 1])
            rs = T(row_sum[h + 1])
            for s in chunk_start:chunk_end  # <= 32 iterations
                prob = exp(scores[h + 1, s] - rm) / rs
                acc += prob * v_cache[v_off + d + 1, s]
            end
            @inbounds attn_out_partial[d + 1, h + 1, c + 1] = acc
        end
    end
end

function fused_attention_forward!(attn_out::AbstractArray{T,1}, q::AbstractArray{T,1},
                                  k_cache::AbstractArray{T,2}, v_cache::AbstractArray{T,2},
                                  n_heads::Int, head_dim::Int, seq_len::Int, n_groups::Int,
                                  chunk_size::Int=32) where T
    n_total = n_heads * head_dim
    n_total == 0 && return attn_out
    if seq_len <= 0
        fill!(attn_out, zero(T))
        return attn_out
    end
    n_chunks = cdiv(head_dim, chunk_size)
    scores_partial = oneAPI.oneArray(zeros(T, n_heads, seq_len, n_chunks))
    kfn = fused_attention_scores_pass1!(_GPU_BACKEND)
    kfn(scores_partial, q, k_cache, n_heads, head_dim, seq_len, n_groups, n_chunks;
        ndrange=(n_heads * seq_len * n_chunks,))
    oneAPI.oneL0.synchronize()

    # Host: reduce partial scores into logits (h, s), then stable softmax stats.
    # NOTE: scores_dev is the RAW LOGITS — pass2 applies exp(x - max)/sum on-device,
    # so the softmax matrix computed below is kept only for host stats, not uploaded.
    scores = reshape(sum(Array(scores_partial), dims=3)[:] / sqrt(T(head_dim)), n_heads, seq_len)
    scores_dev = oneAPI.oneArray(scores)                      # raw logits (h, seq)
    row_max = maximum(scores, dims=2)[:]                      # (n_heads,)
    scores_exp = exp.(scores .- row_max)
    row_sum = sum(scores_exp, dims=2)[:]                      # (n_heads,)
    row_max_dev = oneAPI.oneArray(row_max)
    row_sum_dev = oneAPI.oneArray(row_sum)

    n_chunks_s = cdiv(seq_len, chunk_size)
    attn_out_partial = oneAPI.oneArray(zeros(T, head_dim, n_heads, n_chunks_s))
    kfn = fused_attention_weighted_sum_pass!(_GPU_BACKEND)
    kfn(attn_out_partial, scores_dev, v_cache, n_heads, head_dim, seq_len, n_groups, row_max_dev, row_sum_dev,
        n_chunks_s; ndrange=(n_total * n_chunks_s,))
    oneAPI.oneL0.synchronize()

    copyto!(attn_out, oneAPI.oneArray(sum(Array(attn_out_partial), dims=3)[:]))
    return attn_out
end

end # module GPUCommon
