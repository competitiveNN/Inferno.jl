#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

# Load CPU model
model, tok = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Reset states
Inferno.ModelCPU.reset_states_cpu!(model)

# Create caches
config = model.config
caches = [Inferno.ModelCPU.init_kv_cache_cpu(config, 512) for _ in 1:config.num_hidden_layers]

# Use proper chat format: <|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n
# But for now, let's just use <|im_start|>assistant
# The token for <|im_start|> is 248045 and <|im_end|> is 248046

# Get embeddings for <|im_start|>
im_start_emb = model.embed[:, 248046]  # <|im_start|> is at index 248045 + 1 = 248046
println("Embedding for <|im_start|>:")
println("  norm: ", round(norm(im_start_emb), digits=4))

# Process through all layers
x = im_start_emb
for (i, layer) in enumerate(model.layers)
    global x
    x = layer(x, 0, model.rope, caches[i])
end

# Apply final norm
x_normed = model.final_norm(x)

# Compute logits
logits = model.lm_head * x_normed

# Get tokens
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Top tokens
sorted_idx = sortperm(logits, rev=true)
println("\nTop 20 tokens after <|im_start|>:")
for i in 1:20
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(round(logits[idx], digits=3)) token='$(tokens[idx])'")
end
