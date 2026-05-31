# GPU matmul benchmark — compares naive vs tiled matmul kernels on Intel Arc
using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index, @localmem, @synchronize
using Printf
using LinearAlgebra

devs = oneAPI.devices()
oneAPI.device!(devs[1])
println("Device: $(devs[1])")
println()

# ============================================================
# Naive matmul: one thread per output row
# ============================================================
@kernel function matmul_naive!(y, A, x, n::Int, m::Int)
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

# ============================================================
# Improved matmul: vectorized inner loop (no local memory)
# Reduces loop overhead
# ============================================================
@kernel function matmul_vec4!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    if i <= n
        T = eltype(y)
        acc = zero(T)
        j = 1
        while j + 3 <= m
            @inbounds acc += A[i, j]   * x[j]
            @inbounds acc += A[i, j+1] * x[j+1]
            @inbounds acc += A[i, j+2] * x[j+2]
            @inbounds acc += A[i, j+3] * x[j+3]
            j += 4
        end
        for k in j:m
            @inbounds acc += A[i, k] * x[k]
        end
        @inbounds y[i] = acc
    end
end

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

# ============================================================
# Benchmark suite
# ============================================================
matmuls = [
    ("QKV proj", 2048, 1024),
    ("Gate+Up",  7168, 1024),
    ("Down",     1024, 3584),
    ("SSM in",   2048, 1024),
    ("SSM out",  1024, 2048),
    ("lm_head", 151936, 1024),
]

for (name, n, m) in matmuls
    println("─── $name ($n × $m) Float16 ───")
    
    A = oneArray(Float16.(rand(Float32, n, m)))
    x = oneArray(Float16.(rand(Float32, m)))
    y_ref = Array(A) * Array(x)
    
    # For large matrices, only the fast variants matter
    var_to_eval = [
        ("vec4",  matmul_vec4!),
        ("vec8",  matmul_vec8!),
        ("naive", matmul_naive!),
    ]
    
    for (vname, vkfn) in var_to_eval
        kfn = vkfn(oneAPIBackend())
        y = oneArray(zeros(Float16, n))
        
        kfn(y, A, x, n, m; ndrange=(n,))
        oneAPI.oneL0.synchronize()
        
        t0 = time_ns()
        for _ in 1:5
            kfn(y, A, x, n, m; ndrange=(n,))
            oneAPI.oneL0.synchronize()
        end
        t = (time_ns() - t0) / 5 / 1e6
        
        y_cpu = Array(y)
        err = norm(y_cpu - y_ref) / norm(y_ref)
        
        @printf("  %10s: %7.3fms  err=%.2e\n", vname, t, err)
    end
    println()
end