#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check the attention layer Q/K norm
attn_layer = model.layers[4].op

println("Q norm:")
println("  weight shape: ", size(attn_layer.q_norm.weight))
println("  weight first 5: ", attn_layer.q_norm.weight[1:5])

println("\nK norm:")
println("  weight shape: ", size(attn_layer.k_norm.weight))
println("  weight first 5: ", attn_layer.k_norm.weight[1:5])

# The weight shape should be (head_dim,) for per-head normalization
# But let's see what shape we have

# Test with a matrix input
x = randn(Float32, 256, 8)  # (head_dim, n_heads)
x_normed = attn_layer.q_norm(x)

println("\nTest normalization:")
println("  Input shape: ", size(x))
println("  Output shape: ", size(x_normed))

# Check if normalization is per-column or whole matrix
# Per-column: each column should have the same norm
# Whole matrix: the entire matrix should have unit norm
println("\n  Norm of each column:")
for h in 1:2
    col_norm = norm(view(x_normed, :, h))
    println("    Head $h: ", round(col_norm, digits=3))
end

println("\n  Norm of entire matrix: ", round(norm(x_normed), digits=3))
