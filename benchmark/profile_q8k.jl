#!/usr/bin/env julia
# Profile: how much time does Q8_K quantization take vs the actual dot product?
using Pkg
Pkg.activate(dirname(@__DIR__))

using Inferno
using Inferno.QuantsCPU
using Inferno.QuantizedKernels
using LinearAlgebra

# Load quantized model to get a real matrix
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)

# Get a Q5_K matrix for testing
qmat = model.layers[2].op.in_proj  # Q5_K
println("Matrix: $(typeof(qmat)) outer=$(qmat.outer_dim) inner=$(qmat.inner_dim)")

x = randn(Float32, qmat.inner_dim)
out = zeros(Float32, qmat.outer_dim)

# Time full C kernel
t_full = 0.0
for _ in 1:10
    t_full += @elapsed mul_quant_mat_vec(qmat, x, out)
end
t_full /= 10

# Time just the Q8_K quantization step
q8_data = zeros(UInt8, 2 + 4 + qmat.inner_dim)  # minimal buffer
t_quant = 0.0
for _ in 1:10
    t_quant += @elapsed QuantizedKernels.quantize_row_q8_K_ref!(q8_data, x, qmat.inner_dim)
end
t_quant /= 10

println("Full C GEMV: $(round(t_full*1e3; digits=3))ms")
println("Q8_K quantize only: $(round(t_quant*1e3; digits=3))ms")
println("Quantize fraction: $(round(t_quant/t_full*100; digits=1))%")
