#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Check specific token IDs
for id in [221, 12, 333, 4, 319, 14, 561, 562, 199, 64, 329]
    println("Token $id = '$(tokens[id])'")
end
