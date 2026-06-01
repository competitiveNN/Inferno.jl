#!/usr/bin/env julia
using Inferno
using Inferno.GGUF
using LinearAlgebra
using Statistics

# Load CPU model and get tokenizer
cpu_model, tok = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

println("tokenizer type: ", typeof(tok))

# Test generation with CPU model
println("CPU Model Generation Test:")
println("=" ^ 50)

# Test with a simple prompt
prompt = "The capital of France is"
println("\nPrompt: ", prompt)
println("\nGenerating...")

# Generate using the tokenizer directly
result = stream_to_stdout_cpu(cpu_model, tok, prompt; max_tokens=20, temperature=0.7f0)
println("\n\nDone.")
