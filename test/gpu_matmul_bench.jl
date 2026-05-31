#!/usr/bin/env julia --project=.
# GPU matmul benchmark for Qwen3.5 0.8B

using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using Printf

@kernel function matmul_vec_kernel!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    if i <= n
        T = eltype(y)
        acc = zero(T)
        for j in 1:m
            @inbounds acc += A[i, j] * x[j]
        end
        @inbounds y[i] = acc
    end
end

devs = oneAPI.devices()
oneAPI.device!(devs[1])
kfn = matmul_vec_kernel!(oneAPIBackend())

matmuls = [
    ("embed_lookup", 151936, 1024, 0),
    ("attn_qkv", 2048, 1024, 6),
    ("attn_out", 1024, 2048, 6),
    ("mlp_gateup", 7168, 1024, 24),
    ("mlp_down", 1024, 3584, 24),
    ("ssm_in", 2048, 1024, 18),
    ("ssm_gate", 2048, 1024, 18),
    ("ssm_out", 1024, 2048, 18),
    ("lm_head", 151936, 1024, 1),
]

println("GPU matmul benchmark (Float16) on Qwen3.5 0.8B:")
total = 0.0
for (name, n, m, count) in matmuls
    A = oneArray(Float16.(rand(Float32, n, m)))
    x = oneArray(Float16.(rand(Float32, m)))
    y = oneArray(zeros(Float16, n))

    kfn(y, A, x, n, m; ndrange=(n,))
    oneAPI.oneL0.synchronize()

    t0 = time_ns()
    for _ in 1:5
        kfn(y, A, x, n, m; ndrange=(n,))
        oneAPI.oneL0.synchronize()
    end
    local t = (time_ns() - t0) / 5 / 1e6

    local wt = t * count
    global total += wt
    @printf("  %-15s %6d×%-5d ×%2d = %7.3fms ×%d = %7.2fms\n", name, n, m, count, t, count, wt)
end

println("──────────────────────────────────────────")
@printf("  TOTAL                      per token = %7.2fms\n", total)
@printf("  ESTIMATED TOKENS/SEC       %.1f\n", 1000/total)