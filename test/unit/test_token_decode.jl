#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
tokens = file.metadata["tokenizer.ggml.tokens"]

# Check specific token IDs that were in the top 20
println("Token decoding check:")
for id in [221, 12, 333, 4, 319, 14, 2973, 199, 64, 329, 95794, 25, 27, 2, 272]
    tok = tokens[id]
    println("  id=$id: '$(tok)' (repr: $(repr(tok)))")
end

# The issue is that tokens like id=221 might be multi-byte UTF-8 characters
# Let's check their raw bytes
println("\n\nRaw bytes for token 221:")
tok = tokens[221]
println("  String: '$(tok)'")
println("  Bytes: ", codeunits(tok))
println("  Length: ", length(tok), " chars, ", sizeof(tok), " bytes")
