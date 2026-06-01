#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check if lm_head is tied to embedding
println("Embedding shape: ", size(model.embed))
println("LM head shape: ", size(model.lm_head))

# Check if they share the same data (tied embeddings)
embed_norm = norm(model.embed)
lm_head_norm = norm(model.lm_head)
println("\nEmbedding norm: ", round(embed_norm, digits=3))
println("LM head norm: ", round(lm_head_norm, digits=3))

# For tied embeddings, lm_head = embed'
# Let's check
if size(model.embed, 1) == size(model.lm_head, 2) && size(model.embed, 2) == size(model.lm_head, 1)
    println("\nShapes are compatible for tied embeddings (embed' = lm_head)")
    # Check if they're actually tied
    diff = norm(model.embed' - model.lm_head)
    println("Difference between embed' and lm_head: ", round(diff, digits=6))
end

# Check the actual forward pass
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

println("\nHidden state before final_norm:")
println("  Norm: ", round(norm(x), digits=3))
println("  First 5: ", round.(x[1:5], digits=4))

println("\nHidden state after final_norm:")
println("  Norm: ", round(norm(x_normed), digits=3))
println("  First 5: ", round.(x_normed[1:5], digits=4))

# Compute logits
logits = model.lm_head * x_normed
println("\nLogits:")
println("  Shape: ", size(logits))
println("  Mean: ", round(mean(logits), digits=3))
println("  Std: ", round(std(logits), digits=3))

# The logits should be centered around 0 with std ~2-3 for a well-trained model
