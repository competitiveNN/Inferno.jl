#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Look for thinking-related tokens
println("Looking for 'think' related tokens:")
for (i, tok) in enumerate(tokens)
    tok_str = String(tok)
    if occursin("think", lowercase(tok_str)) || occursin("<|", tok_str) || occursin("start", lowercase(tok_str))
        println("  Token $i (id=$(i-1)): '$(tok)'")
    end
end

# Also check for special tokens in metadata
println("\nSpecial tokens from metadata:")
for (k, v) in file.metadata
    if occursin("token", lowercase(String(k))) || occursin("bos", lowercase(String(k))) || occursin("eos", lowercase(String(k)))
        println("  $k: $v")
    end
end
