#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra

# Load GGUF directly
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check WQ for layer 3 (attention layer)
prefix = "blk.3"
wq_raw = Inferno.LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_q.weight")

println("WQ raw tensor:")
println("  Shape: ", size(wq_raw))
println("  Type: ", eltype(wq_raw))

# GGUF convention: weights are stored as (in_features, out_features)
# For WQ: in_features = hidden_size = 1024, out_features = n_heads * head_dim * 2 = 4096
# But raw shape is (1024, 4096), which is (in_features, out_features)
# We need W of shape (out_features, in_features) for y = W * x
# So we need to transpose: W = raw'

println("\nExpected:")
println("  For y = W * x where x is (hidden,) = (1024,)")
println("  W should be (out_features, hidden) = (4096, 1024)")
println("  Raw is (1024, 4096), transposed is (4096, 1024) ✓")

# Check the values
wq_transposed = wq_raw'
println("\nFirst row of transposed WQ:")
println("  ", round.(wq_transposed[1, 1:5], digits=4))

println("\nFirst column of raw WQ:")
println("  ", round.(wq_raw[1, 1:5], digits=4))

# They should be different since we're looking at different parts of the matrix
# Let's verify by comparing with the model
model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
attn = model.layers[4].op

println("\nLoaded WQ first row:")
println("  ", round.(attn.wq[1, 1:5], digits=4))

println("\nDifference between loaded and transposed:")
diff = attn.wq - wq_transposed
println("  Max diff: ", round(maximum(abs.(diff)), digits=6))
println("  Norm of diff: ", round(norm(diff), digits=3))
