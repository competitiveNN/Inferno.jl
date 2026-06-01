using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get embedding
    h = copy(model.embed[:, 760])
    println("Embedding: norm=$(sqrt(sum(abs2, h)))")

    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Process layer by layer, checking residuals
    for (i, layer) in enumerate(model.layers)
        x_before = copy(h)
        h = layer(h, 0, model.rope, caches[i])

        # Check residual magnitude
        residual = h - x_before
        println("Layer $i ($(layer.is_ssm ? "SSM" : "Attn")): norm=$(sqrt(sum(abs2, h))), residual_norm=$(sqrt(sum(abs2, residual)))")

        if i >= 3
            break
        end
    end
end

main()
