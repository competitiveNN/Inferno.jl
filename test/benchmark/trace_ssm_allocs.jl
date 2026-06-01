#!/usr/bin/env julia
"""Trace allocations in SSM forward pass"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU

gguf_path = "test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

tokens = Inferno.Tokenizer.encode(tokenizer, "Hello")
caches = [ModelCPU.init_kv_cache_cpu(model.config, 256) for _ in model.layers]
x = model.embed[:, tokens[1]]

ssm_layer = model.layers[1].op

println("\n=== Tracing SSM allocations ===\n")

# Test individual operations
println("1. in_proj * x:")
@timev ssm_layer.in_proj * x

println("\n2. gate_proj * x:")
@timev ssm_layer.gate_proj * x

println("\n3. ssm_out * y (simulated):")
y = zeros(Float32, ssm_layer.d_inner)
@timev ssm_layer.ssm_out * y

println("\n4. Full SSM forward:")
@timev ssm_layer(x, 0, model.rope, caches[1])

# Check matrix sizes
println("\n=== Matrix Sizes ===")
println("in_proj: ", size(ssm_layer.in_proj))
println("gate_proj: ", size(ssm_layer.gate_proj))
println("ssm_out: ", size(ssm_layer.ssm_out))
println("ssm_alpha_weight: ", size(ssm_layer.ssm_alpha_weight))
println("ssm_beta_weight: ", size(ssm_layer.ssm_beta_weight))
println("h (state): ", size(ssm_layer.h))
