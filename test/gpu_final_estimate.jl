# GPU matmul estimate for Qwen3.5 0.8B — FINAL with vec8 kernel
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
    println("Device: $(devs[1])\n")
    
    H, IM = 1024, 3584
    N_HEADS, N_KV, HD = 8, 2, 256
    VOCAB = 151936
    N_ATTN, N_SSM, N_LAYERS = 6, 18, 24
    qkv_dim = N_HEADS * HD + 2 * N_KV * HD
    
    matmuls = [
        ("QKV",     qkv_dim, H,  N_ATTN,    ""),
        ("AttnOut", H,       qkv_dim, N_ATTN, ""),
        ("GateUp",  2*IM,    H,  N_LAYERS,  ""),
        ("Down",    H,       IM, N_LAYERS,  ""),
        ("SSMIn",   2048,    H,  N_SSM,     " (sigmoid/bias in matmul)"),
        ("SSMGate", 2048,    H,  N_SSM,     ""),
        ("SSMOut",  H,       2048,  N_SSM,  ""),
        ("LMHead",  VOCAB,   H,  1,         ""),
    ]
    
    kfn = matmul_vec8!(oneAPIBackend())
    
    println("Qwen3.5 0.8B Float16 — vec8 kernel (B580)\n")
    total = 0.0
    for (name, n, m, layers, note) in matmuls
        A = oneArray(Float16.(rand(Float32, n, m)))
        x = oneArray(Float16.(rand(Float32, m)))
        y = oneArray(zeros(Float16, n))
        
        kfn(y, A, x, n, m; ndrange=(n,))
        oneAPI.oneL0.synchronize()
        
        t0 = time_ns()
        for _ in 1:10
            kfn(y, A, x, n, m; ndrange=(n,))
            oneAPI.oneL0.synchronize()
        end
        t = (time_ns() - t0) / 10 / 1e6
        
        per_layer = t * layers
        total += per_layer
        gflops = round(2 * n * m / 1e9 / (t / 1000), digits=1)
        
        @printf("  %-10s %5d×%-5d ×%2d = %7.3fms ×%2d = %7.2fms  %5.1f GFLOPS%s\n",
                name, n, m, layers, t, layers, per_layer, gflops, note)
    end
    
    # Element-wise ops (RMSNorm, SiLU, etc.) — estimate ~10% of matmul time
    elem_ops = total * 0.10
    total_elem = total + elem_ops
    
    println()
    @printf("  Element-wise (est)                         = %7.2fms\n", elem_ops)
    @printf("  ─────────────────────────────────────────────────────\n")
    @printf("  TOTAL                                       = %7.2fms\n", total_elem)
    @printf("  THROUGHPUT                                  = %.1f tok/s\n", 1000 / total_elem)
    println()
    println("For context: CPU achieves 14-18 tok/s with Float32.")
    println("GPU matmul kernel is naive (no shared memory, no oneMKL).")
    println("With working oneMKL: estimated 50-80 tok/s.")
end

main()