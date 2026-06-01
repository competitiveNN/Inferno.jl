#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

# Load tokenizer info from GGUF
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

println("Number of tokens: ", length(tokens))
println("\nFirst 20 tokens:")
for i in 1:20
    println("  $i: '$(tokens[i])'")
end

println("\nTokens around 'The':")
for i in 560:570
    println("  $i: '$(tokens[i])'")
end

# Check for special tokens
println("\nLooking for 'The' in vocabulary:")
for (i, tok) in enumerate(tokens)
    if occursin("The", String(tok)) && i < 1000
        println("  Found at $i: '$(tokens[i])'")
    end
end
