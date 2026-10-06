#!/usr/bin/env julia
# flash_attention_verification.jl - Verify correctness of Flash Attention

using LinearAlgebra
using Printf

include("../src/FlashAttention.jl")

"""
Standard reference attention implementation (exact same math, no optimizations)
"""
function standard_attention!(output::Vector{Float32},
    Q::Vector{Float32},
    cache_k::Array{Float32,3},
    cache_v::Array{Float32,3},
    kv_h::Int,
    seq_len::Int,
    scale::Float32,
    head_dim::Int)

    K_h = view(cache_k, :, kv_h, 1:seq_len)
    V_h = view(cache_v, :, kv_h, 1:seq_len)

    scores = Vector{Float32}(undef, seq_len)
    
    # Compute attention scores
    for i in 1:seq_len
        s = zero(Float32)
        for j in 1:head_dim
            s += K_h[j, i] * Q[j]
        end
        scores[i] = s * scale
    end
    
    # Softmax
    max_score = maximum(scores)
    for i in 1:seq_len
        scores[i] = exp(scores[i] - max_score)
    end
    sum_scores = sum(scores)
    for i in 1:seq_len
        scores[i] /= sum_scores
    end
    
    # Weighted sum
    for i in 1:head_dim
        s = zero(Float32)
        for j in 1:seq_len
            s += V_h[i, j] * scores[j]
        end
        output[i] = s
    end
end

function verify_flash_attention(seq_len::Int; atol::Float32=1.0f-5)
    head_dim = 256
    n_kv_heads = 2
    kv_h = 1
    
    # Setup
    Q = rand(Float32, head_dim)
    cache_k = rand(Float32, head_dim, n_kv_heads, seq_len)
    cache_v = rand(Float32, head_dim, n_kv_heads, seq_len)
    scale = Float32(1.0 / sqrt(head_dim))
    
    output_std = Vector{Float32}(undef, head_dim)
    output_fa = Vector{Float32}(undef, head_dim)
    
    # Compute both
    standard_attention!(output_std, Q, cache_k, cache_v, kv_h, seq_len, scale, head_dim)
    flash_attention_cpu!(output_fa, Q, cache_k, cache_v, kv_h, seq_len, scale, head_dim)
    
    # Compare
    diff = maximum(abs.(output_std .- output_fa))
    relative_diff = diff / maximum(abs.(output_std))
    
    return diff, relative_diff, output_std, output_fa
end

println("="^70)
println("Flash Attention Verification")
println("="^70)
println()

# Random seed for reproducibility
using Random
Random.seed!(42)

seq_lens = [32, 64, 128, 256, 512, 1024, 2048]

println("Seq_len | Max Abs Diff | Relative Diff | Status")
println("-"^70)

all_passed = true
for seq_len in seq_lens
    diff, rel_diff, std_out, fa_out = verify_flash_attention(seq_len, atol=Float32(1e-4))
    passed = diff < Float32(1e-4)
    global all_passed &= passed
    status = passed ? "PASS" : "FAIL"
    
    @printf "  %4d  |   %.6e  |   %.6e   | %s\n" seq_len diff rel_diff status
    
    if !passed
        println("  Sample outputs:")
        println("    Standard[1:5]: ", std_out[1:5])
        println("    FlashAttn[1:5]: ", fa_out[1:5])
    end
end

println()
if all_passed
    println("✓ All tests PASSED!")
    println("Flash Attention produces numerically equivalent output to standard attention.")
else
    println("✗ Some tests FAILED!")
    println("Differences may indicate numerical issues.")
end
println("="^70)
