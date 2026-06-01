#!/usr/bin/env julia
"""Test CPU-only generation after attention optimization"""

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))

using Inferno
using Inferno.GGUF
using Inferno.LoaderCPU
using Inferno.ModelCPU

println("Loading GGUF model (CPU only)...")
gguf_path = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
model, tokenizer = load_model_cpu(gguf_path)

prompt = "What is 2 + 2 ?"
tokens = Inferno.Tokenizer.encode(tokenizer, prompt)
println("Prompt: ", prompt)
println("Tokens: ", tokens)

# Generate 20 tokens using CPU
println("\nGenerating 20 tokens...")
caches = [ModelCPU.init_kv_cache_cpu(model.config, 256) for _ in model.layers]

# Process prompt first
local logits
logits = ModelCPU.forward_cpu!(model, tokens, 0, caches)
local pos
pos = length(tokens)

# Generate tokens
generated_tokens = Int[]
for i in 1:20
    # Get last logits
    next_token = argmax(logits[:, end])
    push!(generated_tokens, next_token)
    
    # Forward with new token
    global logits = ModelCPU.forward_cpu!(model, [next_token], pos, caches)
    global pos = pos + 1
end

# Decode
output = Inferno.Tokenizer.decode(tokenizer, generated_tokens)
println("\nGenerated tokens: ", generated_tokens)
println("Decoded output: ", output)

# Compare with expected
if occursin("4", output)
    println("\n✓ PASS: Output contains '4'")
else
    println("\n✗ FAIL: Output does not contain '4'")
end
