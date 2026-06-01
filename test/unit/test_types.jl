using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test forward pass
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    h = copy(view(model.embed, :, 561))
    println("After embed: type=", typeof(h), " size=", size(h))

    for (i, layer) in enumerate(model.layers)
        h = layer(h, 0, model.rope, caches[i])
        println("After layer $i: type=", typeof(h), " size=", size(h))
    end

    println("\nBefore final_norm: type=", typeof(h))
    h_normed = model.final_norm(h)
    println("After final_norm: type=", typeof(h_normed))
end

main()
