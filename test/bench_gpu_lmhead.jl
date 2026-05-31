# Tiled lm_head matmul kernel — splits (248320, 1024) matmul into chunks
# Run: julia --project=. test/bench_gpu_lmhead.jl <model.gguf>

using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using LinearAlgebra, Printf

@info "Loading Inferno..."
using Inferno
using Inferno.ModelCPU

# ── Tiled matmul: process rows in blocks to avoid GPU timeout ──
@kernel function matmul_block!(y, A, x, n, m, row_offset)
    # Each thread handles one row of the full matrix
    i_global = @index(Global, Linear) + row_offset
    if i_global <= size(A, 1)
        acc = zero(Float32)
        j = 1
        while j + 7 <= m
            @inbounds acc += A[i_global, j+0] * x[j+0] + A[i_global, j+1] * x[j+1] +
                              A[i_global, j+2] * x[j+2] + A[i_global, j+3] * x[j+3] +
                              A[i_global, j+4] * x[j+4] + A[i_global, j+5] * x[j+5] +
                              A[i_global, j+6] * x[j+6] + A[i_global, j+7] * x[j+7]
            j += 8
        end
        for k in j:m
            @inbounds acc += A[i_global, k] * x[k]
        end
        @inbounds y[i_global] = acc
    end
end
const _MBLOCK = matmul_block!(oneAPIBackend())

function gpu_lm_head!(output, weight, input; block_size=10240)
    n_rows, n_cols = size(weight)
    @assert length(input) == n_cols
    @assert length(output) == n_rows
    
    for start_row in 1:block_size:n_rows
        end_row = min(start_row + block_size - 1, n_rows)
        actual = end_row - start_row + 1
        
        # Launch kernel for this block
        _MBLOCK(output, weight, input, actual, n_cols, start_row - 1; ndrange=(actual,))
        oneAPI.oneL0.synchronize()
    end
end

function main()
    gguf = ARGS[1]
    devs = oneAPI.devices(); oneAPI.device!(devs[1])
    @info "Device: $(devs[1])"

    @info "Loading model..."
    cmodel, tok = Inferno.load_model_cpu(gguf)
    cfg = cmodel.config; h = cfg.hidden_size
    @info "Config: $(cfg.vocab_size) vocab, $h hidden"

    # Upload lm_head to GPU
    @info "Uploading lm_head ($(size(cmodel.lm_head)))..."
    lm_head_gpu = oneArray{Float32}(cmodel.lm_head)

    # Benchmark input
    x_cpu = rand(Float32, h)
    x_gpu = oneArray{Float32}(x_cpu)
    out_gpu = oneArray{Float32}(undef, cfg.vocab_size)

    # Warmup with small block
    @info "Warming up (block size 10240)..."
    gpu_lm_head!(out_gpu, lm_head_gpu, x_gpu; block_size=1024)
    @info "Warmup done."

    # Benchmark with different block sizes
    for bs in [4096, 8192, 16384, 32768, 65536]
        out_gpu .= 0.0f0
        t0 = time()
        gpu_lm_head!(out_gpu, lm_head_gpu, x_gpu; block_size=bs)
        t = time() - t0
        @printf("  block_size=%6d: %.1fms\n", bs, t * 1000)

        # Verify vs CPU
        cpu_out = cmodel.lm_head * x_cpu
        GPU_out = Array(out_gpu)
        sim = dot(cpu_out, GPU_out) / (norm(cpu_out) * norm(GPU_out))
        @printf("    cosim=%.6f  max_diff=%.6f\n", sim, maximum(abs.(cpu_out - GPU_out)))
    end

    # Benchmark single-block (no tiling) for comparison
    @info "Single block (no tiling, hangs test)..."
    out_gpu .= 0.0f0
    t0 = time()
    gpu_lm_head!(out_gpu, lm_head_gpu, x_gpu; block_size=cfg.vocab_size)
    t = time() - t0
    @printf("  single block: %.1fms\n", t * 1000)
end

main()