using Inferno
using Statistics

function main()
    cpu_model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    gpu_model, _ = load_model("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Get embedding for token 760 "The"
    cpu_embed = copy(view(cpu_model.embed, :, 760))
    gpu_embed = Float32.(view(gpu_model.embed, :, 760))

    println("Embedding comparison:")
    println("  CPU mean: ", mean(cpu_embed), " std: ", std(cpu_embed))
    println("  GPU mean: ", mean(gpu_embed), " std: ", std(gpu_embed))
    println("  Max diff: ", maximum(abs.(cpu_embed - gpu_embed)))

    # Forward pass
    caches = [ModelCPU.init_kv_cache_cpu(cpu_model.config, 512) for _ in 1:cpu_model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(cpu_model)

    h = copy(cpu_embed)
    for (i, layer) in enumerate(cpu_model.layers)
        h = layer(h, 0, cpu_model.rope, caches[i])
    end

    h_normed = cpu_model.final_norm(h)
    cpu_logits = cpu_model.lm_head * h_normed

    println("\nCPU logits: mean=", mean(cpu_logits), " std=", std(cpu_logits))
    println("CPU top5: ", sortperm(cpu_logits, rev=true)[1:5])

    # Check Paris token
    paris_token = 11751  # ĠParis
    println("\n' Paris' logit (CPU): ", cpu_logits[paris_token])
end

main()
