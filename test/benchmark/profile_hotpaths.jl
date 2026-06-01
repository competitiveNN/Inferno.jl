#!/usr/bin/env julia
"""Profile CPU inference to identify hot paths"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using Profile
using Printf

println("Loading GGUF model...")
gguf_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

prompt = "What is 2 + 2 ?"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Warm up
caches = [ModelCPU.init_kv_cache_cpu(model.config, 256) for _ in model.layers]
_ = ModelCPU.forward_cpu!(model, tokens, 0, caches)

# Profile
println("\nProfiling forward pass...")
Profile.init(n=10000)
Profile.clear()

# Run multiple forward passes to collect profile data
for i in 1:100
    Profile.@profile ModelCPU.forward_cpu!(model, tokens, 0, caches)
end

# Print profile
println("\n=== Profile Results ===")
Profile.print(format=:flat, sortedby=:count, mincount=10)

# Also profile token generation
println("\n\n=== Profiling Token Generation ===")
Profile.clear()

# Generate some tokens
pos = length(tokens)
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)

for i in 1:50
    next_token = argmax(logits[:, end])
    Profile.@profile begin
        logits = ModelCPU.forward_cpu!(model, [next_token], pos, caches)
    end
    pos += 1
end

println("\n=== Generation Profile ===")
Profile.print(format=:flat, sortedby=:count, mincount=5)
