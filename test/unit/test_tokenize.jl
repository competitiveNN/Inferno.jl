#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using Inferno.Tokenizer

# Load GGUF
file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Create a BPETokenizer
tok = load_tokenizer(file.metadata)

# Test tokenization - BPETokenizer uses 1-indexed IDs
text = "The"
ids = encode(tok, text)
println("Tokenizing '$text':")
println("  Token IDs: ", ids)
for id in ids
    println("  Token $id: '$(tok.id_to_token[id])'")
end

# Test decode
decoded = decode(tok, ids)
println("\nDecoded: '$decoded'")

# Compare with manual lookup
println("\nManual token lookup for 'The':")
for (i, t) in enumerate(tokens)
    if t == "The" || t == "ĠThe" || occursin("The", t)
        println("  Found at index $i: '$(t)'")
        if i > 600
            break
        end
    end
end
