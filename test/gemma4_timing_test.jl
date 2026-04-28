#!/usr/bin/env julia
# Gemma4 — Proper timing test with fresh state per generation
using Pkg; Pkg.activate(".")
using Printf
using Inferno

function main()
    model, tok = Inferno.Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it")
    config = model.config
    
    encode_fn(t, s) = Inferno.Gemma4Loader.encode(t, s)
    decode_fn(ids) = Inferno.Gemma4Loader.decode(tok, ids)
    
    # Test single prompt with proper timing
    prompt = "What is 2 + 2 ?"
    println("\n=== Greedy Generation (temperature=0) ===")
    
    result = Inferno.Gemma4.generate_text_gemma4(model, tok, prompt;
        max_tokens=64,
        temperature=0.0f0,
        top_k=0,
        repetition_penalty=1.0f0,
        encode_fn=encode_fn,
        decode_fn=decode_fn)
    println("\nResult: ", result)
    
    # Test second prompt — check if shared_kv state leaks
    println("\n=== Second prompt (should NOT be instant) ===")
    result2 = Inferno.Gemma4.generate_text_gemma4(model, tok, "The capital of France is";
        max_tokens=64,
        temperature=0.0f0,
        top_k=0,
        repetition_penalty=1.0f0,
        encode_fn=encode_fn,
        decode_fn=decode_fn)
    println("\nResult: ", result2)
    
    println()
end

main()
