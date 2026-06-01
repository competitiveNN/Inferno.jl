using Inferno
using Statistics

function main()
    file = Inferno.GGUF.read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Manually encode "The capital of France is"
    prompt_tokens = [561, 6511, 314, 9338, 284]
    println("Prompt tokens: ", prompt_tokens)

    # Decode to verify
    decoded = join([replace(tokens[t+1], "Ġ" => " ") for t in prompt_tokens])
    println("Decoded: ", decoded)

    # Now test model with these tokens
    model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Run forward pass
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process prompt tokens (except last one)
    for (i, tok) in enumerate(prompt_tokens[1:end-1])
        h = copy(view(model.embed, :, tok))
        for (j, layer) in enumerate(model.layers)
            h = layer(h, i-1, model.rope, caches[j])
        end
    end

    # Generate from last token
    last_tok = prompt_tokens[end]
    h = copy(view(model.embed, :, last_tok))
    for (j, layer) in enumerate(model.layers)
        h = layer(h, length(prompt_tokens)-1, model.rope, caches[j])
    end
    h = model.final_norm(h)
    logits = model.lm_head * h

    println("\nLogits stats: mean=", mean(logits), " std=", std(logits), " min=", minimum(logits), " max=", maximum(logits))

    # Top 10 tokens
    top10 = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 next tokens:")
    for t in top10
        println("  Token $t: ", repr(tokens[t+1]), " (logit=", logits[t], ")")
    end
end

main()
