# GPU matmul estimate for Qwen3.5 0.8B using vec8 matmul kernel
using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using Printf

@kernel function matmul_vec8!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    if i <= n
        T = eltype(y)
        acc = zero(T)
        j = 1
        while j + 7 <= m
            @inbounds for k in 0:7
                acc += A[i, j+k] * x[j+k]
            end
            j += 8
        end
        for k in j:m
            @inbounds acc += A[i, k] * x[k]
        end
        @inbounds y[i] = acc
    end
end

function main()
    devs = oneAPI.devices()
    oneAPI.device!(devs[1])
    println("Device: $(devs[1])")
    
    kfn = matmul_vec8!(oneAPIBackend())
    
    H = 1024
    IM = 3584
    N_HEADS = 8
    N_KV = 2
    HD = 256
    VOCAB = 151936
    N_ATTN = 6
    N_SSM = 18
    N_LAYERS = 24
    
    qkv_dim = N_HEADS * HD + 2 * N_KV * HD
    
    matmuls = [
        ("QKV",     qkv_dim, H,     N_ATTN),
        ("AttnOut", H,       qkv_dim, N_ATTN),
        ("GateUp",  2*IM,    H,     N_LAYERS),
        ("Down",    H,       IM,    N_LAYERS),
        ("SSMIn",   2048,    H,     N_SSM),
        ("SSMGate", 2048,    H,     N_SSM),
        ("SSMOut",  H,       2048,  N_SSM),
        ("LMHead",  VOCAB,   H,     1),
    ]
    
    println("Benchmarking Qwen3.5 0.8B matmuls (Float16, vec8 kernel)...\n")
    
    total_ms = 0.0
    for (name, n, m, layers) in matmuls
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
        t_ms = (time_ns() - t0) / 5 / 1e6

        gflop = 2 * n * m / 1e9
        gflops = gflop / (t_ms / 1000)
        per_layer = t_ms * layers
        total_ms += per_layer

        @printf("  %-10s %5d×%-5d ×%2d = %7.3fms ×%2d = %7.2fms  (%.1f GFLOPS)\n",
                name, n, m, layers, t_ms, layers, per_layer, gflops)
    end
    
    println()
    @printf("Total GPU forward: %.2f ms per token\n", total_ms)
    @printf("Throughput:        %.1f tokens/sec\n", 1000 / total_ms)
end

main()