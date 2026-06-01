#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Check the top tokens
println("Top predicted tokens:")
for id in [221, 333, 2973, 12, 4, 319, 25, 329, 199, 95794]
    println("  Token $id (id=$(id-1)): '$(tokens[id])'")
end

println("\n\nSpecial tokens:")
println("  Token 248046 (id=248045): '<|im_start|>'")
println("  Token 248047 (id=248046): '<|im_end|>'")
println("  Token 248069 (id=248068): 'ölker'")
println("  Token 248070 (id=248069): ' illustrate'")
