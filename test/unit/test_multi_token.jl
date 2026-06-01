using Inferno
using Statistics
using LinearAlgebra

function main()
    model, file = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Initialize
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process two tokens: " The" (561) and " capital" (6511)
    prompt_tokens = [561, 6511]

    # Process first token
    x = copy(view(model.embed, :, prompt_tokens[1]))
    for i in 1:3
        x = model.layers[i](x, 0, model.rope, caches[i])
    end

    # Now process second token through attention at pos=1
    x = model.layers[4](x, 0, model.rope, caches[4])  # Note: pos should be 1, but cache is at pos 0

    println("After layer 4 with pos=0: mean=$(mean(x)), std=$(std(x))")

    # Reset and try with pos=1
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process first token
    x1 = copy(view(model.embed, :, prompt_tokens[1]))
    for i in 1:3
        x1 = model.layers[i](x1, 0, model.rope, caches[i])
    end
    x1 = model.layers[4](x1, 0, model.rope, caches[4])

    # Process second token
    x2 = copy(view(model.embed, :, prompt_tokens[2]))
    for i in 1:3
        x2 = model.layers[i](x2, 1, model.rope, caches[i])
    end

    println("\nProcessing second token at pos=1:")
    println("Before layer 4: mean=$(mean(x2)), std=$(std(x2))")

    # Check RoPE application at pos=1
    attn = model.layers[4].op
    h = model.layers[4].in_norm(x2)
    qkv = attn.wq * h
    q_size = attn.n_heads * attn.head_dim
    query = reshape(qkv[1:q_size], attn.head_dim, attn.n_heads)
    query_normed = attn.q_norm(query)

    println("\nBefore RoPE (pos=1):")
    println("  query[1:4, 1]: ", query_normed[1:4, 1])

    ModelCPU.apply_rotary_emb!(query_normed, 1, model.rope)

    println("\nAfter RoPE (pos=1):")
    println("  query[1:4, 1]: ", query_normed[1:4, 1])

    # Full forward
    x2 = model.layers[4](x2, 1, model.rope, caches[4])
    println("\nAfter layer 4: mean=$(mean(x2)), std=$(std(x2))")

    # Continue through all layers
    x2 = model.layers[5](x2, 1, model.rope, caches[5])
    println("After layer 5: mean=$(mean(x2)), std=$(std(x2))")

    # Full pass through all layers
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    for (pos, tok) in enumerate(prompt_tokens)
        h = copy(view(model.embed, :, tok))
        for (i, layer) in enumerate(model.layers)
            h = layer(h, pos-1, model.rope, caches[i])
        end
    end

    # Get logits
    h = model.final_norm(h)
    logits = model.lm_head * h

    println("\n=== Logits after 2 tokens ===")
    println("mean=$(mean(logits)), std=$(std(logits))")
    top5 = sortperm(logits, rev=true)[1:5]
    for t in top5
        println("  Token $t: ", repr(tokens[t+1]), " (logit=", logits[t], ")")
    end
end

main()
