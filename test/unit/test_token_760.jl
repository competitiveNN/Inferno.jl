#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Create caches
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Forward pass for "The" (token 760, which is 0-indexed, so we use 761 in 1-indexed)
x = model.embed[:, 761]  # "The" (no space)
println("Input: token 761 'The'")

for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, 0, model.rope, caches[i])
end

x_normed = model.final_norm(x)
logits = model.lm_head * x_normed

# Top tokens
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]
sorted_idx = sortperm(logits, rev=true)

println("\nTop 20 tokens:")
for i in 1:20
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end
