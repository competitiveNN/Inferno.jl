#!/usr/bin/env julia
# Profile quantized C kernels vs F32 BLAS for each quant type
using Pkg
Pkg.activate(dirname(@__DIR__))

using Inferno
using Inferno.QuantsCPU
using Inferno.QuantizedKernels
using Inferno.ModelCPU: mul_quant_mat_vec
using LinearAlgebra
using BenchmarkTools

# Load both models
model_q, tok = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
model_f, _   = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

# Collect representative matrices of each quant type
q4_mat = nothing; q5_mat = nothing; q6_mat = nothing; q8_mat = nothing
f4_mat = nothing; f5_mat = nothing; f6_mat = nothing; f8_mat = nothing

for (i, layer) in enumerate(model_q.layers)
    if layer.is_ssm
        ssm = layer.ssm
        if isa(ssm.in_proj, Q4_K_Matrix) && q4_mat === nothing
            q4_mat = ssm.in_proj
            # Find corresponding F32
            f4_mat = model_f.layers[i].ssm.in_proj
            println("Q4_K in_proj: rows=$(q4_mat.rows), cols=$(q4_mat.cols), inner=$(q4_mat.inner_dim), blocks=$(q4_mat.n_blocks)")
        end
        if isa(ssm.gate_proj, Q5_K_Matrix) && q5_mat === nothing
            q5_mat = ssm.gate_proj
            f5_mat = model_f.layers[i].ssm.gate_proj
            println("Q5_K gate_proj: rows=$(q5_mat.rows), cols=$(q5_mat.cols), inner=$(q5_mat.inner_dim), blocks=$(q5_mat.n_blocks)")
        end
        if isa(ssm.ssm_out, Q6_K_Matrix) && q6_mat === nothing
            q6_mat = ssm.ssm_out
            f6_mat = model_f.layers[i].ssm.ssm_out
            println("Q6_K ssm_out: rows=$(q6_mat.rows), cols=$(q6_mat.cols), inner=$(q6_mat.inner_dim), blocks=$(q6_mat.n_blocks)")
        end
        if isa(ssm.ssm_out, Q8_0_Matrix) && q8_mat === nothing
            q8_mat = ssm.ssm_out
            f8_mat = model_f.layers[i].ssm.ssm_out
            println("Q8_0 ssm_out: rows=$(q8_mat.rows), cols=$(q8_mat.cols), inner=$(q8_mat.inner_dim), blocks=$(q8_mat.n_blocks)")
        end
    else
        attn = layer.attention
        if isa(attn.o_proj, Q8_0_Matrix) && q8_mat === nothing
            q8_mat = attn.o_proj
            f8_mat = model_f.layers[i].attention.o_proj
            println("Q8_0 o_proj: rows=$(q8_mat.rows), cols=$(q8_mat.cols), inner=$(q8_mat.inner_dim), blocks=$(q8_mat.n_blocks)")
        end
    end
end

println("\n=== Benchmarking GEMV: C kernel vs F32 BLAS ===")

# Helper: benchmark a quantized matvec
function bench_quant(name, qmat, fmat, x_dim)
    x = randn(Float32, x_dim)
    out_q = zeros(Float32, qmat.rows)
    out_f = zeros(Float32, qmat.rows)
    
    # Quantized C kernel
    t_q = @belapsed mul_quant_mat_vec($qmat, $x, $out_q) seconds=2
    
    # F32 BLAS
    t_f = @belapsed mul!($out_f, $fmat, $x) seconds=2
    
    ratio = t_q / t_f
    println("  $name: C_kernel=$(round(t_q*1e3; digits=2))ms  BLAS=$(round(t_f*1e3; digits=2))ms  ratio=$(round(ratio; digits=2))x  $(ratio > 1 ? "SLOWER" : "FASTER")")
end

if q4_mat !== nothing
    bench_quant("Q4_K", q4_mat, f4_mat, q4_mat.cols)
end

if q5_mat !== nothing
    bench_quant("Q5_K", q5_mat, f5_mat, q5_mat.cols)
end

if q6_mat !== nothing
    bench_quant("Q6_K", q6_mat, f6_mat, q6_mat.cols)
end

if q8_mat !== nothing
    bench_quant("Q8_0", q8_mat, f8_mat, q8_mat.cols)
end
