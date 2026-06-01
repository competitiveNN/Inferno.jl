#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

println("GGUF Metadata:")
for (k, v) in file.metadata
    if occursin("ssm", lowercase(String(k))) || occursin("rope", lowercase(String(k))) || occursin("sliding", lowercase(String(k)))
        println("  $k: $v")
    end
end
