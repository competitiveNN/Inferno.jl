# GPU matmul benchmark — optimized variants for B580
using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using Printf
using LinearAlgebra: norm

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

# Multi-row per thread (handles 2 rows)
@kernel function matmul_2row!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    i2 = 2 * i
    if i2 <= n
        T = eltype(y)
        acc1 = zero(T)
        acc2 = zero(T)
        for j in 1:m
            @inbounds begin
                acc1 += A[i2-1, j] * x[j]
                acc2 += A[i2,   j] * x[j]
            end
        end
        @inbounds begin
            y[i2-1] = acc1
            y[i2]   = acc2
        end
    elseif i2 - 1 <= n
        acc = zero(T)
        for j in 1:m
            @inbounds acc += A[i2-1, j] * x[j]
        end
        @inbounds y[i2-1] = acc
    end
end

function main()
    devs = oneAPI.devices()
    oneAPI.device!(devs[1])
    println("Device: $(devs[1])\n")
    
    H = 1024
    IM = 3584
    N_HEADS = 8
    N_KV = 2
    HD = 256
    VOCAB = 151936
    
    qkv_dim = N_HEADS * HD + 2 * N_KV * HD
    
    matmuls = [
        ("QKV",     qkv_dim, H),
        ("AttnOut", H,       qkv_dim),
        ("GateUp",  2*IM,    H),
        ("Down",    H,       IM),
        ("LMHead",  VOCAB,   H),
    ]
    
    kfn_vec8 = matmul_vec8!(oneAPIBackend())
    kfn_2row = matmul_2row!(oneAPIBackend())
    
    println("Optimized matmul comparison (Float16)\n")
    for (name, n, m) in matmuls
        A_gpu = oneArray(Float16.(rand(Float32, n, m)))
        x_gpu = oneArray(Float16.(rand(Float32, m)))
        y_ref = Array(A_gpu) * Array(x_gpu)
        
        A = A_gpu
        x = x_gpu
        
        results = []
        
        for (kname, kfn, nd) in [
            ("vec8",  kfn_vec8, n),
            ("2row",  kfn_2row, n ÷ 2 + 1),  # half the threads for 2 rows each
        ]
            y = oneArray(zeros(Float16, n))
            kfn(y, A, x, n, m; ndrange=(nd,))
            oneAPI.oneL0.synchronize()
            
            t0 = time_ns()
            for _ in 1:10
                kfn(y, A, x, n, m; ndrange=(nd,))
                oneAPI.oneL0.synchronize()
            end
            t = (time_ns() - t0) / 10 / 1e6
            
            y_cpu = Array(y)
            err = norm(y_cpu - y_ref) / norm(y_ref)
            
            push!(results, (kname, t, err))
        end
        
        @printf("  %-10s %5d×%-5d\n", name, n, m)
        for (kname, t, err) in results
            @printf("    %10s: %7.3fms  err=%.2e\n", kname, t, err)
        end
    end
end

main()