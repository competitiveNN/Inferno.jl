#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Get embedding for <|im_start|> (token 248045)
im_start_id = 248046  # 1-indexed
emb = model.embed[:, im_start_id]

println("Julia: Embedding for <|im_start|> (id=$(im_start_id-1)):")
println("  shape: ", size(emb))
println("  norm: ", round(sqrt(sum(abs2, emb)), digits=4))
println("  first 5: ", emb[1:5])
