#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Check for BOS/EOS tokens
println("Looking for special tokens:")
for (i, tok) in enumerate(tokens)
    tok_str = String(tok)
    if occursin("<|im_start|>", tok_str) || occursin("<|im_end|>", tok_str) || occursin("<|endoftext|>", tok_str)
        println("  Token $(i-1): '$(tok)'")
    end
end

# Check for the chat template tokens
println("\nChat template tokens:")
println("  <|im_start|> at index: ", findfirst(x -> x == "<|im_start|>", tokens))
println("  <|im_end|> at index: ", findfirst(x -> x == "<|im_end|>", tokens))

# Check what token 248045 and 248046 are
println("\nSpecial token indices:")
println("  Token 248045: '$(tokens[248046])'")
println("  Token 248046: '$(tokens[248047])'")
