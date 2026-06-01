#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Load embedding
embed_data = Inferno.LoaderCPU.extract_tensor_cpu(file, "token_embd.weight")
println("Embed shape: ", size(embed_data))

# Load final norm
final_norm_w = Inferno.LoaderCPU.extract_tensor_cpu(file, "output_norm.weight")
println("Final norm shape: ", size(final_norm_w))
println("Final norm values: first 5 = ", final_norm_w[1:5])

# Use "The" token (id=561)
the_id = 561
x = embed_data[:, the_id+1]  # 1-indexed
println("\nEmbedding for 'The' (id=$the_id):")
println("  norm: ", norm(x))
println("  first 5: ", x[1:5])

# Apply RMS norm manually
ss = sum(x.^2)
m = ss / length(x)
scale = 1.0 / sqrt(m + 1e-6)
x_normed = x .* scale .* vec(final_norm_w)

println("\nAfter final norm:")
println("  norm: ", norm(x_normed))
println("  first 5: ", x_normed[1:5])

# LM head (tied to embedding)
logits = embed_data' * x_normed
println("\nLogits:")
println("  shape: ", size(logits))
println("  max: ", maximum(logits))
println("  min: ", minimum(logits))

# Top tokens
sorted_idx = sortperm(logits, rev=true)
println("\nTop 10 tokens (direct from embedding):")
for i in 1:10
    idx = sorted_idx[i]
    println("  $i: id=$idx logit=$(logits[idx]) token='$(tokens[idx])'")
end
