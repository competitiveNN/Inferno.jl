using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test with just "The" token
    token = 760
    pos = 0

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    h = copy(model.embed[:, token])
    println("Embed: norm=$(sqrt(sum(abs2, h)))")

    # Process through all layers
    for (i, layer) in enumerate(model.layers)
        h_before = copy(h)
        h = layer(h, pos, model.rope, caches[i])
        
        # Track gradient-like behavior (how much does h change)
        delta = h - h_before
        
        if i % 4 == 0 || i == length(model.layers)
            println("Layer $i ($(layer.is_ssm ? "SSM" : "Attn")): norm=$(sqrt(sum(abs2, h))), delta_norm=$(sqrt(sum(abs2, delta)))")
        end
    end

    # Final norm
    h_normed = model.final_norm(h)
    println("\nFinal norm: norm=$(sqrt(sum(abs2, h_normed)))")

    # Logits
    logits = model.lm_head * h_normed
    println("Logits: norm=$(sqrt(sum(abs2, logits)))")

    # Softmax to get probabilities
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)
    
    println("\nTop 10 by probability:")
    top10 = sortperm(probs, rev=true)[1:10]
    tokens = file.metadata["tokenizer.ggml.tokens"]
    for t in top10
        println("  Token $t: $(repr(tokens[t+1])) (prob=$(probs[t]), logit=$(logits[t]))")
    end
end

main()
