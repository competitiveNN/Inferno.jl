#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Load GGUF directly
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Compare layer 4 attention weights
attn = model.layers[4].op
prefix = "blk.3"

println("Layer 4 (index 3) attention weights comparison:")

# WQ
wq_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_q.weight")
wq_loaded = attn.wq
println("\nWQ:")
println("  Raw shape: ", size(wq_raw))
println("  Loaded shape: ", size(wq_loaded))
println("  Raw first 5: ", round.(wq_raw[1:5], digits=4))
println("  Loaded first 5: ", round.(wq_loaded[1:5, 1], digits=4))

# WK
wk_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_k.weight")
wk_loaded = attn.wk
println("\nWK:")
println("  Raw shape: ", size(wk_raw))
println("  Loaded shape: ", size(wk_loaded))
println("  Raw first 5: ", round.(wk_raw[1:5], digits=4))
println("  Loaded first 5: ", round.(wk_loaded[1:5, 1], digits=4))

# WO
wo_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_output.weight")
wo_loaded = attn.wo
println("\nWO:")
println("  Raw shape: ", size(wo_raw))
println("  Loaded shape: ", size(wo_loaded))
println("  Raw first 5: ", round.(wo_raw[1:5], digits=4))
println("  Loaded first 5: ", round.(wo_loaded[1:5, 1], digits=4))

# Check Q norm
q_norm_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_q_norm.weight")
println("\nQ norm:")
println("  Raw shape: ", size(q_norm_raw))
println("  Loaded shape: ", size(attn.q_norm.weight))
println("  Raw first 5: ", round.(q_norm_raw[1:5], digits=4))
println("  Loaded first 5: ", round.(attn.q_norm.weight[1:5], digits=4))
