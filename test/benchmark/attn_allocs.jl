#!/usr/bin/env julia
"""Count allocations in attention forward pass"""

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

# Get first attention layer (layer 4, 0-indexed = layer 3)
local attn_layer
for (i, layer) in enumerate(model.layers)
    if !layer.is_ssm
        global attn_layer = layer.op
        println("Attention layer found at index $i (1-indexed)")
        break
    end
end

println("Attention layer type: ", typeof(attn_layer))

prompt = "Hello"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Warm up
caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
x = model.embed[:, tokens[1]]
_ = attn_layer(x, 0, model.rope, caches[4])

# Benchmark
println("\nBenchmarking attention layer forward pass:")
b = @benchmark attn_layer($x, 0, $(model.rope), $(caches[4]))
show(stdout, MIME("text/plain"), b)
println()

println("\nMemory allocated per call: ", b.memory, " bytes")
println("Allocations per call: ", b.allocs)
