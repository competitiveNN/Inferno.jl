#!/usr/bin/env julia
# Direct GEMV benchmark: C kernel vs BLAS for each quant type
using Pkg
Pkg.activate(dirname(@__DIR__))

using Inferno
using Inferno.QuantsCPU
using Inferno.QuantizedKernels
using Inferno.ModelCPU: mul_quant_mat_vec
using LinearAlgebra

function main()
    # Load both models
    println("Loading quantized model...")
    model_q, tok = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=true)
    println("Loading F32 model...")
    model_f, _   = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"; keep_quantized=false)

    # Collect one of each quant type from ANY field
    q4_qm = nothing; q4_fm = nothing
    q5_qm = nothing; q5_fm = nothing
    q6_qm = nothing; q6_fm = nothing
    q8_qm = nothing; q8_fm = nothing

    for (i, layer) in enumerate(model_q.layers)
        op = layer.op
        mlp = layer.mlp
        
        # Check all quantized weights in this layer
        all_qmats = Dict{String, Any}()
        all_fmats = Dict{String, Any}()
        
        if layer.is_ssm
            all_qmats["in_proj"] = op.in_proj
            all_qmats["gate_proj"] = op.gate_proj
            all_qmats["ssm_out"] = op.ssm_out
            all_fmats["in_proj"] = model_f.layers[i].op.in_proj
            all_fmats["gate_proj"] = model_f.layers[i].op.gate_proj
            all_fmats["ssm_out"] = model_f.layers[i].op.ssm_out
        end
        
        all_qmats["gate_w"] = mlp.gate_weight
        all_qmats["up_w"] = mlp.up_weight
        all_qmats["down_w"] = mlp.down_weight
        all_fmats["gate_w"] = model_f.layers[i].mlp.gate_weight
        all_fmats["up_w"] = model_f.layers[i].mlp.up_weight
        all_fmats["down_w"] = model_f.layers[i].mlp.down_weight
        
        for (fname, qm) in all_qmats
            fm = all_fmats[fname]
            if isa(qm, Q4_K_Matrix) && q4_qm === nothing
                q4_qm = qm; q4_fm = fm
                println("Q4_K L$i $fname: outer=$(qm.outer_dim) inner=$(qm.inner_dim)")
            end
            if isa(qm, Q5_K_Matrix) && q5_qm === nothing
                q5_qm = qm; q5_fm = fm
                println("Q5_K L$i $fname: outer=$(qm.outer_dim) inner=$(qm.inner_dim)")
            end
            if isa(qm, Q6_K_Matrix) && q6_qm === nothing
                q6_qm = qm; q6_fm = fm
                println("Q6_K L$i $fname: outer=$(qm.outer_dim) inner=$(qm.inner_dim)")
            end
            if isa(qm, Q8_0_Matrix) && q8_qm === nothing
                q8_qm = qm; q8_fm = fm
                println("Q8_0 L$i $fname: outer=$(qm.outer_dim) inner=$(qm.inner_dim)")
            end
        end
        
        # Stop once we have all 4
        if q4_qm !== nothing && q5_qm !== nothing && q6_qm !== nothing && q8_qm !== nothing
            break
        end
    end

    println("\n=== GEMV Benchmark: C kernel vs F32 BLAS (10 runs each) ===")

    function bench_gemv(name, qmat, fmat)
        x = randn(Float32, qmat.inner_dim)
        out_q = zeros(Float32, qmat.outer_dim)
        out_f = zeros(Float32, size(fmat, 1))
        
        # Warmup
        for _ in 1:3
            mul_quant_mat_vec(qmat, x, out_q)
            mul!(out_f, fmat, x)
        end
        
        # Time C kernel
        t_q = 0.0
        for _ in 1:10
            t_q += @elapsed mul_quant_mat_vec(qmat, x, out_q)
        end
        t_q /= 10
        
        # Time F32 BLAS
        t_f = 0.0
        for _ in 1:10
            t_f += @elapsed mul!(out_f, fmat, x)
        end
        t_f /= 10
        
        ratio = t_q / t_f
        speedup = ratio > 1 ? "SLOWER ($(round(ratio; digits=2))x)" : "FASTER ($(round(1/ratio; digits=2))x)"
        println("  $name ($(qmat.outer_dim)x$(qmat.inner_dim)): C=$(round(t_q*1e3; digits=3))ms  BLAS=$(round(t_f*1e3; digits=3))ms  $speedup")
    end

    if q4_qm !== nothing; bench_gemv("Q4_K", q4_qm, q4_fm); end
    if q5_qm !== nothing; bench_gemv("Q5_K", q5_qm, q5_fm); end
    if q6_qm !== nothing; bench_gemv("Q6_K", q6_qm, q6_fm); end
    if q8_qm !== nothing; bench_gemv("Q8_0", q8_qm, q8_fm); end
end

main()
