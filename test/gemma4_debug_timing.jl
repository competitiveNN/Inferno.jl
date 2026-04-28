#!/usr/bin/env julia
# Gemma4 — Debug timing: correct shared KV handling
using Pkg; Pkg.activate(".")
using Printf
using Inferno

function main()
    model, tok = Inferno.Gemma4Loader.load_gemma4("test/models/gemma-4-E2B-it")
    config = model.config
    
    encode_fn(t, s) = Inferno.Gemma4Loader.encode(t, s)
    decode_fn(ids) = Inferno.Gemma4Loader.decode(tok, ids)
    
    # Build prompt tokens manually
    chat_prompt = "<|turn>user\nWhat is 2 + 2 ?<turn|><|turn>model\n"
    prompt_tokens = encode_fn(tok, chat_prompt)
    if isempty(prompt_tokens) || prompt_tokens[1] != 3
        pushfirst!(prompt_tokens, 3)
    end
    println("Prompt tokens: ", length(prompt_tokens), " tokens")
    
    # Manual forward pass with timing (DO NOT clear shared_kv within generation!)
    cache = Inferno.Gemma4.init_kv_cache(config, 4096)
    empty!(model.shared_kv_k)
    empty!(model.shared_kv_v)
    
    t0 = time()
    logits = Inferno.Gemma4.forward!(model, prompt_tokens, 0, cache)
    t1 = time()
    println("\nPrefill: $(round((t1-t0)*1000; digits=1)) ms for $(length(prompt_tokens)) tokens")
    
    # Sample first token
    next_token = argmax(logits)
    println("First token: $next_token = '$(decode_fn([next_token]))'")
    
    # Generate tokens — shared KV should persist WITHIN this generation
    curr_pos = length(prompt_tokens)
    last_token = next_token
    stop_tokens = Set{Int}([107, 2]) # <turn|>, EOS
    total_gen_time = 0.0
    for i in 1:64
        t0 = time()
        logits = Inferno.Gemma4.forward!(model, [last_token], curr_pos, cache)
        t1 = time()
        dt = (t1 - t0) * 1000
        total_gen_time += dt
        curr_pos += 1
        next_token = argmax(logits)
        tok_str = decode_fn([next_token])
        print(tok_str)
        flush(stdout)
        if next_token in stop_tokens
            break
        end
        last_token = next_token
    end
    println()
    gen_count = curr_pos - length(prompt_tokens)
    println("\nGeneration: $gen_count tokens in $(round(total_gen_time; digits=1)) ms = $(round(gen_count / (total_gen_time/1000); digits=1)) tok/s")
    
    # Second generation with FRESH cache and cleared shared KV
    println("\n=== Second prompt ===")
    cache2 = Inferno.Gemma4.init_kv_cache(config, 4096)
    empty!(model.shared_kv_k)
    empty!(model.shared_kv_v)
    
    chat_prompt2 = "<|turn>user\nThe capital of France is<turn|><|turn>model\n"
    prompt_tokens2 = encode_fn(tok, chat_prompt2)
    if isempty(prompt_tokens2) || prompt_tokens2[1] != 3
        pushfirst!(prompt_tokens2, 3)
    end
    
    t0 = time()
    logits2 = Inferno.Gemma4.forward!(model, prompt_tokens2, 0, cache2)
    t1 = time()
    println("Prefill: $(round((t1-t0)*1000; digits=1)) ms for $(length(prompt_tokens2)) tokens")
    
    next_token = argmax(logits2)
    print(decode_fn([next_token]))
    flush(stdout)
    
    curr_pos = length(prompt_tokens2)
    last_token = next_token
    total_gen_time = 0.0
    for i in 1:64
        t0 = time()
        logits2 = Inferno.Gemma4.forward!(model, [last_token], curr_pos, cache2)
        t1 = time()
        dt = (t1 - t0) * 1000
        total_gen_time += dt
        curr_pos += 1
        next_token = argmax(logits2)
        tok_str = decode_fn([next_token])
        print(tok_str)
        flush(stdout)
        if next_token in stop_tokens
            break
        end
        last_token = next_token
    end
    println()
    gen_count = curr_pos - length(prompt_tokens2)
    println("\nGeneration: $gen_count tokens in $(round(total_gen_time; digits=1)) ms = $(round(gen_count / (total_gen_time/1000); digits=1)) tok/s")
    
    println()
end

main()
