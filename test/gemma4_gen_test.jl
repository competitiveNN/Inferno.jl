#!/usr/bin/env julia
# Gemma4 CPU Inference — Comprehensive Test
# Tests: greedy generation, sampling, multiple prompts
using Pkg; Pkg.activate(".")
using Printf
using Inferno

function main()
    model, tok = Inferno.Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it")
    config = model.config

    encode_fn(t, s) = Inferno.Gemma4Loader.encode(t, s)
    decode_fn(ids) = Inferno.Gemma4Loader.decode(tok, ids)

    println("\n=== Greedy Generation Tests ===")
    prompts = [
        "What is 2 + 2 ?",
        "The capital of France is",
        "What is the largest planet in our solar system?",
    ]
    
    for prompt in prompts
        result = Inferno.Gemma4.generate_text_gemma4(model, tok, prompt;
            max_tokens=128,
            temperature=0.0f0,
            top_k=0,
            repetition_penalty=1.0f0,
            encode_fn=encode_fn,
            decode_fn=decode_fn)
        println("\n  Q: $prompt")
        println("  A: $result")
    end

    println("\n\n=== Sampling Generation Test (temp=0.7, top_k=40) ===")
    result = Inferno.Gemma4.generate_text_gemma4(model, tok, "Explain what recursion is in simple terms.";
        max_tokens=128,
        temperature=0.7f0,
        top_k=40,
        repetition_penalty=1.1f0,
        encode_fn=encode_fn,
        decode_fn=decode_fn)
    println("\n  Q: Explain what recursion is in simple terms.")
    println("  A: $result")
    
    println()
end

main()
