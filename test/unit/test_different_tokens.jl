using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Process different tokens and check output
    test_tokens = [
        (760, "The"),
        (6511, " capital"),
        (11751, "Paris"),
    ]

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    for (tok_id, tok_name) in test_tokens
        # Reset state
        ModelCPU.reset_states_cpu!(model)
        for c in caches
            fill!(c.k, 0)
            fill!(c.v, 0)
        end

        h = copy(model.embed[:, tok_id])
        for (i, layer) in enumerate(model.layers)
            h = layer(h, 0, model.rope, caches[i])
        end

        h_normed = model.final_norm(h)
        logits = model.lm_head * h_normed

        top5 = sortperm(logits, rev=true)[1:5]

        println("\nToken '$tok_name' ($tok_id):")
        println("  Hidden norm after model: $(sqrt(sum(abs2, h)))")
        println("  Top 5 predictions:")
        for t in top5
            println("    $(repr(tokens[t+1])) (logit=$(logits[t]))")
        end
    end
end

main()
