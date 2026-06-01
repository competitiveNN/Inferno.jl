#!/usr/bin/env julia
"""Count allocations in SSM forward pass"""

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

# Get first SSM layer
ssm_layer = model.layers[1].op
println("SSM layer type: ", typeof(ssm_layer))

prompt = "Hello"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Warm up
caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
x = model.embed[:, tokens[1]]
_ = ssm_layer(x, 0, model.rope, caches[1])

# Benchmark allocations
println("\nBenchmarking SSM layer forward pass:")
b = @benchmark ssm_layer($x, 0, $(model.rope), $(caches[1]))
show(stdout, MIME("text/plain"), b)
println()

println("\nMemory allocated per call: ", b.memory)
