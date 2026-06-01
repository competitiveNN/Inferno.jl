#!/usr/bin/env julia
"""Benchmark individual layer components to identify hot paths"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using BenchmarkTools
using Printf

println("Loading GGUF model...")
gguf_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

prompt = "Hello"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Setup
caches = [ModelCPU.init_kv_cache_cpu(model.config, 256) for _ in model.layers]
x = model.embed[:, tokens[1]]

println("\n=== Benchmarking Individual Layers ===\n")

# Benchmark each layer type
ssm_times = Float64[]
attn_times = Float64[]
mlp_times = Float64[]

for (i, layer) in enumerate(model.layers)
    if layer.is_ssm
        b = @benchmark $layer.op($x, 0, $(model.rope), $(caches[i]))
        push!(ssm_times, median(b).time / 1000)  # microseconds
        println("Layer $i (SSM): $(round(median(b).time/1000, digits=1)) μs")
    else
        b = @benchmark $layer.op($x, 0, $(model.rope), $(caches[i]))
        push!(attn_times, median(b).time / 1000)
        println("Layer $i (Attn): $(round(median(b).time/1000, digits=1)) μs")
    end
end

# Benchmark MLP separately
for (i, layer) in enumerate(model.layers)
    if layer.is_ssm
        b = @benchmark $layer.mlp($x)
        push!(mlp_times, median(b).time / 1000)
        break  # Just first MLP
    end
end

println("\n=== Summary ===")
println("SSM layers: $(length(ssm_times))x")
println("  Avg time: $(round(sum(ssm_times)/length(ssm_times), digits=1)) μs")
println("  Total: $(round(sum(ssm_times), digits=1)) μs")

println("\nAttention layers: $(length(attn_times))x")
println("  Avg time: $(round(sum(attn_times)/length(attn_times), digits=1)) μs")
println("  Total: $(round(sum(attn_times), digits=1)) μs")

println("\nMLP (per layer): $(round(mlp_times[1], digits=1)) μs")
println("  Total (24 layers): $(round(mlp_times[1] * 24, digits=1)) μs")

# Calculate total forward pass time
total_ssm = sum(ssm_times)
total_attn = sum(attn_times)
total_mlp = mlp_times[1] * 24
total_estimate = total_ssm + total_attn + total_mlp

println("\n=== Estimated Forward Pass ===")
println("SSM:  $(round(total_ssm, digits=1)) μs ($(round(total_ssm/total_estimate*100, digits=1))%)")
println("Attn: $(round(total_attn, digits=1)) μs ($(round(total_attn/total_estimate*100, digits=1))%)")
println("MLP:  $(round(total_mlp, digits=1)) μs ($(round(total_mlp/total_estimate*100, digits=1))%)")
println("Total: $(round(total_estimate, digits=1)) μs")

# Benchmark full forward pass
println("\n=== Full Forward Pass ===")
b_full = @benchmark ModelCPU.forward_cpu!($model, $tokens, 0, $caches)
println("Median: $(round(median(b_full).time/1000, digits=1)) μs")
println("Memory: $(b_full.memory ÷ 1024) KiB, $(b_full.allocs) allocs")
