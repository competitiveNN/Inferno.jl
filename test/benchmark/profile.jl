#!/usr/bin/env julia
"""Profile CPU inference to find hot paths"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using Profile
using Printf

println("=== CPU Inference Profiling ===")
println()

# Load model
println("Loading GGUF model...")
gguf_path = "test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

prompt = "What is 2 + 2 ?"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)

# Warm up
caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
_ = ModelCPU.generate_cpu(model, Int.(tokens), 0, caches; temperature=0.0f0)

# Profile
println("\nProfiling 100 token generations...")
Profile.clear()

for i in 1:100
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
    
    Profile.@profile begin
        for j in 1:5
            if j == 1
                ModelCPU.generate_cpu(model, Int.(tokens), 0, caches; temperature=0.0f0)
            else
                ModelCPU.generate_cpu(model, [1], length(tokens) + j - 2, caches; temperature=0.0f0)
            end
        end
    end
end

println("\nTop functions by time spent:")
Profile.print(format=:flat, sortedby=:count, maxdepth=15)

# Save profile to file
open("test/benchmark/profile_results.txt", "w") do f
    Profile.print(IOContext(f, :compact => true), format=:flat, sortedby=:count, maxdepth=30)
end
println("\nFull profile saved to test/benchmark/profile_results.txt")
