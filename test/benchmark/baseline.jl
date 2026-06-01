#!/usr/bin/env julia
"""Baseline performance benchmark for CPU inference"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU
using Printf

println("=== CPU Inference Baseline Benchmark ===")
println()

# Load model
println("Loading GGUF model...")
gguf_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)
println()

# Test prompts of different lengths
test_cases = [
    ("Short (4 tok)", "Hello"),
    ("Medium (8 tok)", "What is 2 + 2?"),
    ("Long (20 tok)", "The quick brown fox jumps over the lazy dog and then"),
    ("Very Long (50 tok)", "In computer science, a hash table is a data structure that implements an associative array abstract data type, a structure that can map keys to values."),
]

println("| Prompt | Tokens | Gen Time (s) | Tokens/Sec | Latency (ms/token) |")
println("|--------|--------|--------------|------------|--------------------|")

for (name, prompt) in test_cases
    tokens = Inferno.Tokenizer.encode(tokenizer, prompt)
    num_prompt = length(tokens)
    
    # Warm up
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
    _ = ModelCPU.generate_cpu(model, Int.(tokens), 0, caches; temperature=0.0f0)
    
    # Benchmark
    num_gen = 32
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 100) for _ in model.layers]
    
    start_time = time()
    generated = Int[]
    for i in 1:num_gen
        if i == 1
            next_token, _ = ModelCPU.generate_cpu(model, Int.(tokens), 0, caches; temperature=0.0f0)
        else
            pos = length(tokens) + i - 2
            next_token, _ = ModelCPU.generate_cpu(model, [generated[end]], pos, caches; temperature=0.0f0)
        end
        push!(generated, next_token)
    end
    elapsed = time() - start_time
    
    tps = num_gen / elapsed
    latency_ms = (elapsed / num_gen) * 1000
    
    @printf("| %s | %d | %.3f | %.2f | %.1f |\n", name, num_prompt, elapsed, tps, latency_ms)
end

println()
println("=== Memory Usage ===")
println("Model size: ", Base.summarysize(model) / 1024 / 1024, " MB")
