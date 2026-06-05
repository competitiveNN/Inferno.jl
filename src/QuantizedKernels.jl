"""
Quantized SIMD Kernels Module

Provides C AVX2 SIMD kernels for quantized matrix-vector multiplication.
Uses the llama.cpp approach: quantize F32 activations to Q8_K (int8) once,
then do integer dot products using _mm256_maddubs_epi16 (32 mul+16 add/instr).

Threading: C kernel uses OpenMP for row-level parallelism.
Set OMP_NUM_THREADS to control C threading (default: uses all cores).
When using quantized kernels, reduce BLAS threads to avoid oversubscription.
"""
module QuantizedKernels

export quant_kernels_available
export q4_k_gemv!, q5_k_gemv!, q6_k_gemv!, q8_0_gemv!
export quantize_q8_k!, vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k

# ============================================================================
# C AVX2 Kernel Loading (compiled from kernels/quant_kernels.c)
# ============================================================================

const LIBQUANT_PATH = joinpath(@__DIR__, "kernels", "libquant_kernels.so")
const _libquant_loaded = Ref(false)
const _force_disabled = Ref{Bool}(false)

function __init__()
    # Allow disabling C quant kernels via env var
    if haskey(ENV, "INFERNO_NO_QUANT_KERNELS") && ENV["INFERNO_NO_QUANT_KERNELS"] != "0"
        _force_disabled[] = true
        @info "C quant kernels disabled by INFERNO_NO_QUANT_KERNELS"
    end
end

function _load_libquant()
 _force_disabled[] && return false
 if !_libquant_loaded[]
 if isfile(LIBQUANT_PATH)
 _libquant_loaded[] = true
 # Set OpenMP threads to match available cores
 n = clamp(Sys.CPU_THREADS ÷ 2, 2, 8)
 ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", string(n))
 elseif !_quant_missing_warned[]
 _quant_missing_warned[] = true
 @debug "Quantized C kernel not found at $LIBQUANT_PATH - falling back to Julia"
 end
 end
 return _libquant_loaded[]
end

"""Check if C SIMD kernels are available (and not disabled by INFERNO_NO_QUANT_KERNELS)."""
quant_kernels_available() = _load_libquant()

# Q8_K block size in bytes: 4 (float d) + 256 (int8_t qs) + 32 (int16_t bsums) = 292
const Q8_K_BLOCK_BYTES = 292

# ============================================================================
# Q8_K Pre-allocated Buffer Pool
# ============================================================================

const _q8_bufs = Dict{Int, Vector{UInt8}}()
const _quant_missing_warned = Ref{Bool}(false)

function get_q8_k_buffer(inner_dim::Int)
    @assert inner_dim % 256 == 0 "Q8_K requires inner_dim divisible by 256, got $inner_dim"
    n_blocks = inner_dim ÷ 256
    buf_size = n_blocks * Q8_K_BLOCK_BYTES
    buf = get!(_q8_bufs, inner_dim) do
        Vector{UInt8}(undef, buf_size)
    end
    @assert length(buf) >= buf_size "Q8_K buffer too small: $(length(buf)) < $buf_size"
    return buf
end

# ============================================================================
# C Kernel Wrappers (OpenMP threading handled inside C)
# ============================================================================

"""
    q4_k_gemv!(out, weight_data, x, inner_dim, outer_dim)

Q4_K matrix-vector multiply using C AVX2 SIMD kernel with OpenMP threading.
Quantizes input to Q8_K once, then computes dot products in parallel.
"""
function q4_k_gemv!(out::Vector{Float32}, weight_data::Vector{UInt8},
                     x::Vector{Float32}, inner_dim::Int, outer_dim::Int)
    _load_libquant() || error("Quantized C kernel not available")
    
    q8_buf = get_q8_k_buffer(inner_dim)
    
    ccall((:q4_k_gemv, LIBQUANT_PATH), Cvoid,
        (Ptr{UInt8}, Ptr{Float32}, Ptr{Float32}, Ptr{UInt8}, Cint, Cint),
        weight_data, x, out, q8_buf, inner_dim, outer_dim)
    
    return out
end

"""
 q5_k_gemv!(out, weight_data, x, inner_dim, outer_dim)

Q5_K matrix-vector multiply using C AVX2 SIMD kernel with OpenMP threading.
"""
function q5_k_gemv!(out::Vector{Float32}, weight_data::Vector{UInt8},
	x::Vector{Float32}, inner_dim::Int, outer_dim::Int)
	_load_libquant() || error("Quantized C kernel not available")
	
	q8_buf = get_q8_k_buffer(inner_dim)
	
	ccall((:q5_k_gemv, LIBQUANT_PATH), Cvoid,
	(Ptr{UInt8}, Ptr{Float32}, Ptr{Float32}, Ptr{UInt8}, Cint, Cint),
	weight_data, x, out, q8_buf, inner_dim, outer_dim)
	
	return out
end

"""
 q6_k_gemv!(out, weight_data, x, inner_dim, outer_dim)

Q6_K matrix-vector multiply using C AVX2 SIMD kernel with OpenMP threading.
Quantizes input to Q8_K once, then computes dot products in parallel.
"""
function q6_k_gemv!(out::Vector{Float32}, weight_data::Vector{UInt8},
	x::Vector{Float32}, inner_dim::Int, outer_dim::Int)
	_load_libquant() || error("Quantized C kernel not available")
	
	q8_buf = get_q8_k_buffer(inner_dim)
	
	ccall((:q6_k_gemv, LIBQUANT_PATH), Cvoid,
	(Ptr{UInt8}, Ptr{Float32}, Ptr{Float32}, Ptr{UInt8}, Cint, Cint),
	weight_data, x, out, q8_buf, inner_dim, outer_dim)
	
	return out
end

"""
    q8_0_gemv!(out, weight_data, x, inner_dim, outer_dim)

Q8_0 matrix-vector multiply using C AVX2 SIMD kernel.
"""
function q8_0_gemv!(out::Vector{Float32}, weight_data::Vector{UInt8},
                     x::Vector{Float32}, inner_dim::Int, outer_dim::Int)
    _load_libquant() || error("Quantized C kernel not available")
    
    ccall((:q8_0_gemv, LIBQUANT_PATH), Cvoid,
        (Ptr{UInt8}, Ptr{Float32}, Ptr{Float32}, Cint, Cint),
        weight_data, x, out, inner_dim, outer_dim)
    
    return out
end

"""Quantize F32 vector to Q8_K format (called internally by gemv functions)."""
function quantize_q8_k!(q8_buf::Vector{UInt8}, x::Vector{Float32}, inner_dim::Int)
    _load_libquant() || error("Quantized C kernel not available")
    ccall((:quantize_row_q8_K, LIBQUANT_PATH), Cvoid,
          (Ptr{UInt8}, Ptr{Float32}, Cint), q8_buf, x, inner_dim)
    return q8_buf
end

"""Single Q4_K block row dot product with Q8_K vector."""
function vec_dot_q4_k_q8_k(weight_ptr::Ptr{UInt8}, q8_ptr::Ptr{UInt8}, nb::Cint)
    _load_libquant() || error("Quantized C kernel not available")
    return ccall((:vec_dot_q4_K_q8_K, LIBQUANT_PATH), Cfloat,
                 (Ptr{UInt8}, Ptr{UInt8}, Cint), weight_ptr, q8_ptr, nb)
end

"""Single Q5_K block row dot product with Q8_K vector."""
function vec_dot_q5_k_q8_k(weight_ptr::Ptr{UInt8}, q8_ptr::Ptr{UInt8}, nb::Cint)
	_load_libquant() || error("Quantized C kernel not available")
	return ccall((:vec_dot_q5_K_q8_K, LIBQUANT_PATH), Cfloat,
	(Ptr{UInt8}, Ptr{UInt8}, Cint), weight_ptr, q8_ptr, nb)
end

"""Single Q6_K block row dot product with Q8_K vector."""
function vec_dot_q6_k_q8_k(weight_ptr::Ptr{UInt8}, q8_ptr::Ptr{UInt8}, nb::Cint)
	_load_libquant() || error("Quantized C kernel not available")
	return ccall((:vec_dot_q6_K_q8_K, LIBQUANT_PATH), Cfloat,
	(Ptr{UInt8}, Ptr{UInt8}, Cint), weight_ptr, q8_ptr, nb)
end

end # module QuantizedKernels
