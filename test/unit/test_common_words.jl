#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Reset all states
Inferno.ModelCPU.reset_states_cpu!(model)
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Get embedding
x = model.embed[:, 761]  # "The"

# Process through all layers
for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, 0, model.rope, caches[i])
end

# Apply final norm
x_normed = model.final_norm(x)

# Compute logits
logits = model.lm_head * x_normed

# Check top tokens
println("Top 30 tokens:")
sorted_idx = sortperm(logits, rev=true)
for i in 1:30
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end

# Check tokens for common English words
println("\n\nLogits for common English words:")
common_words = ["the", "The", "Ġthe", "ĠThe", "Ġa", "Ġto", "Ġand", "Ġof", "Ġin", "Ġis", "Ġthat", "Ġwas", "Ġfor", "Ġon", "Ġare", "Ġwith", "Ġbe", "Ġhave", "Ġthis", "Ġfrom"]
for word in common_words
    # Find the token
    for (j, tok) in enumerate(tokens)
        if tok == word
            println("  '$word' (id=$(j-1)): logit=$(round(logits[j], digits=3))")
            break
        end
    end
end
