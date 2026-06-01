#!/usr/bin/env julia
"""Quick benchmark of layer timing"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Printf
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

println("\n=== Quick Layer Timing (1 sample each) ===\n")

# Quick timing
for (i, layer) in enumerate(model.layers)
    layer_type = layer.is_ssm ? "SSM" : "Attn"
    t = @elapsed layer.op(x, 0, model.rope, caches[i])
    @printf("Layer %d (%s): %.3f ms\n", i, layer_type, t * 1000)
end

# Benchmark embedding lookup
println("\n=== Embedding Lookup ===")
b = @benchmark $model.embed[:, $(tokens[1])]
println("Time: $(round(median(b).time/1000, digits=1)) μs")

# Benchmark final lm_head
println("\n=== LM Head (output projection) ===")
logits = similar(x, model.config.vocab_size)
b = @benchmark mul!($logits, $(model.lm_head), $x)
println("Time: $(round(median(b).time/1000, digits=1)) μs")
println("Memory: $(b.memory) bytes")
