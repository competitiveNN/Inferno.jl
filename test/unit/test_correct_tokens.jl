using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Use correct tokens: "The capital of France is" -> [760, 6511, 314, 9338, 369]
    prompt_tokens = [760, 6511, 314, 9338, 369]

    println("Prompt tokens: ", prompt_tokens)
    println("Decoded: ", join([replace(tokens[t+1], "Ġ" => " ") for t in prompt_tokens]))

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process all tokens
    h = nothing
    for (pos, tok_id) in enumerate(prompt_tokens)
        h = copy(view(model.embed, :, tok_id))
        for (i, layer) in enumerate(model.layers)
            h = layer(h, pos-1, model.rope, caches[i])
        end
    end

    h = model.final_norm(h)
    logits = model.lm_head * h

    println("\nLogits: mean=$(mean(logits)), std=$(std(logits)), min=$(minimum(logits)), max=$(maximum(logits))")
    top10 = sortperm(logits, rev=true)[1:10]
    println("\nTop 10 tokens:")
    for t in top10
        println("  Token $t: ", repr(tokens[t+1]), " (logit=", logits[t], ")")
    end

    # Check where Paris is
    paris_idx = findfirst(==("ĠParis"), tokens)
    if paris_idx !== nothing
        println("\n' Paris' (token $(paris_idx-1)) logit: ", logits[paris_idx-1])
    end
end

main()
