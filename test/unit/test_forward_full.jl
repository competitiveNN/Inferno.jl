#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)

# Create caches
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Forward pass for a single token
x = model.embed[:, 562]  # "ĠThe"
println("=== Forward pass for token 562 'ĠThe' ===")
println("\nInput embedding: norm=$(round(norm(x), digits=4))")

# Process through all layers
for (i, layer) in enumerate(model.layers)
    global x
    x_before = copy(x)
    x = layer(x, 0, model.rope, caches[i])
    diff = norm(x .- x_before)
    
    # Check for NaN or Inf
    if any(isnan.(x)) || any(isinf.(x))
        println("ERROR: Layer $i produced NaN or Inf!")
        break
    end
    
    println("Layer $i ($(layer.is_ssm ? "SSM" : "Attn")): norm=$(round(norm(x), digits=4)), diff=$(round(diff, digits=4))")
end

# Final norm
x_normed = model.final_norm(x)
println("\nAfter final_norm: norm=$(round(norm(x_normed), digits=4))")

# LM head
logits = model.lm_head * x_normed
println("\nLogits: shape=$(size(logits)), range=[$(round(minimum(logits), digits=2)), $(round(maximum(logits), digits=2))]")

# Top tokens
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]
sorted_idx = sortperm(logits, rev=true)

println("\nTop 20 tokens:")
for i in 1:20
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end

# Also check some expected tokens
println("\n\nExpected tokens after 'The':")
for word in ["following", "most", "first", "best", " next", " way", " result", " reason", " idea", " thing"]
    # Find the token
    for (j, tok) in enumerate(tokens)
        if tok == word
            println("  '$word' (id=$j): logit=$(round(logits[j], digits=3))")
            break
        end
    end
end
