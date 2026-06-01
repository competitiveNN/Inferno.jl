#!/usr/bin/env julia
"""Benchmark SSM layer allocations in detail"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using BenchmarkTools

println("Loading GGUF model...")
gguf_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

prompt = "Hello"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Setup
caches = [ModelCPU.init_kv_cache_cpu(model.config, 256) for _ in model.layers]
x = model.embed[:, tokens[1]]

# Get first SSM layer
ssm_layer = model.layers[1].op

println("\n=== SSM Layer Allocation Breakdown ===\n")

# Benchmark individual operations
println("1. in_proj * x:")
b = @benchmark ssm_mat_vec_mul($(ssm_layer.in_proj), $x)
println("   Time: $(round(median(b).time/1000, digits=1)) μs")
println("   Memory: $(b.memory) bytes, $(b.allocs) allocs")

println("\n2. gate_proj * x:")
b = @benchmark ssm_mat_vec_mul($(ssm_layer.gate_proj), $x)
println("   Time: $(round(median(b).time/1000, digits=1)) μs")
println("   Memory: $(b.memory) bytes, $(b.allocs) allocs")

println("\n3. ssm_out * y:")
y = zeros(Float32, ssm_layer.d_inner)
b = @benchmark ssm_mat_vec_mul($(ssm_layer.ssm_out), $y)
println("   Time: $(round(median(b).time/1000, digits=1)) μs")
println("   Memory: $(b.memory) bytes, $(b.allocs) allocs")

println("\n4. Full SSM forward:")
b = @benchmark $ssm_layer($x, 0, $(model.rope), $(caches[1]))
println("   Time: $(round(median(b).time/1000, digits=1)) μs")
println("   Memory: $(b.memory ÷ 1024) KiB, $(b.allocs) allocs")

println("\n=== Attention Layer ===")
attn_layer = model.layers[4].op
b = @benchmark $attn_layer($x, 0, $(model.rope), $(caches[4]))
println("Time: $(round(median(b).time/1000, digits=1)) μs")
println("Memory: $(b.memory ÷ 1024) KiB, $(b.allocs) allocs")

println("\n=== MLP Layer ===")
mlp = model.layers[1].mlp
b = @benchmark $mlp($x)
println("Time: $(round(median(b).time/1000, digits=1)) μs")
println("Memory: $(b.memory ÷ 1024) KiB, $(b.allocs) allocs")
