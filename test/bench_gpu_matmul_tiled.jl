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
# Tiled matmuls with different tile sizes
# ============================================================
@kernel function matmul_tiled_16!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    T = eltype(y)
    lmem = @localmem T (16,)
    if i <= n
        acc = zero(T)
        lid = @index(Local, Linear)
        for tile_start in 1:16:m
            if tile_start + lid - 1 <= m
                @inbounds lmem[lid] = x[tile_start + lid - 1]
            end
            @synchronize
            K = min(16, m - tile_start + 1)
            for k in 1:K
                @inbounds acc += A[i, tile_start + k - 1] * lmem[k]
            end
            @synchronize
        end
        @inbounds y[i] = acc
    end
end

@kernel function matmul_tiled_32!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    T = eltype(y)
    lmem = @localmem T (32,)
    if i <= n
        acc = zero(T)
        lid = @index(Local, Linear)
        for tile_start in 1:32:m
            if tile_start + lid - 1 <= m
                @inbounds lmem[lid] = x[tile_start + lid - 1]
            end
            @synchronize
            K = min(32, m - tile_start + 1)
            for k in 1:K
                @inbounds acc += A[i, tile_start + k - 1] * lmem[k]
            end
            @synchronize
        end
        @inbounds y[i] = acc
    end
end

@kernel function matmul_tiled_64!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    T = eltype(y)
    lmem = @localmem T (64,)
    if i <= n
        acc = zero(T)
        lid = @index(Local, Linear)
        for tile_start in 1:64:m
            if tile_start + lid - 1 <= m
                @inbounds lmem[lid] = x[tile_start + lid - 1]
            end
            @synchronize
            K = min(64, m - tile_start + 1)
            for k in 1:K
                @inbounds acc += A[i, tile_start + k - 1] * lmem[k]
            end
            @synchronize
        end
        @inbounds y[i] = acc
    end
end

@kernel function matmul_tiled_128!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    T = eltype(y)
    lmem = @localmem T (128,)
    if i <= n
        acc = zero(T)
        lid = @index(Local, Linear)
        for tile_start in 1:128:m
            if tile_start + lid - 1 <= m
                @inbounds lmem[lid] = x[tile_start + lid - 1]
            end
            @synchronize
            K = min(128, m - tile_start + 1)
            for k in 1:K
                @inbounds acc += A[i, tile_start + k - 1] * lmem[k]
            end
            @synchronize
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
    
    best_t = Inf
    best_name = ""
    
    for (kname, kfn_factory, ws) in [
        ("tiled_16", matmul_tiled_16!, 16),
        ("tiled_32", matmul_tiled_32!, 32),
        ("tiled_64", matmul_tiled_64!, 64),
        ("tiled_128", matmul_tiled_128!, 128),
    ]
        kfn = kfn_factory(oneAPIBackend(), ws)
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
        
        if t < best_t
            best_t = t
            best_name = kname
        end
        
        @printf("  %10s: %7.3fms  err=%.2e\n", kname, t, err)
    end
    
    # Naive (no workgroup specified)
    kfn_naive = matmul_naive!(oneAPIBackend())
    y = oneArray(zeros(Float16, n))
    kfn_naive(y, A, x, n, m; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    
    t0 = time_ns()
    for _ in 1:5
        kfn_naive(y, A, x, n, m; ndrange=(n,))
        oneAPI.oneL0.synchronize()
    end
    t_naive = (time_ns() - t0) / 5 / 1e6
    
    @printf("  naive:       %7.3fms\n", t_naive)
    @printf("  BEST:  %s  %7.3fms  (%.2fx naive)\n", best_name, best_t, t_naive / best_t)
    println()
end