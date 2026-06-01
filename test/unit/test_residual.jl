using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

    h = copy(model.embed[:, 761])
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Forward through layers 1-6
    for i in 1:6
        layer = model.layers[i]
        h = layer(h, 0, model.rope, caches[i])
    end

    println("Input to layer 7:")
    println("  h norm: ", round(sqrt(sum(abs2, h)), digits=3))

    # Apply layer 7 manually
    layer = model.layers[7]
    ssm = layer.op
    
    # Apply in_norm
    h_norm = layer.in_norm(h)
    println("\nAfter in_norm:")
    println("  norm: ", round(sqrt(sum(abs2, h_norm)), digits=3))

    # Forward through SSM
    ssm_out = ssm(h_norm, 0, model.rope, caches[7])
    println("\nAfter SSM:")
    println("  output norm: ", round(sqrt(sum(abs2, ssm_out)), digits=3))
    
    # Residual connection
    h_after_attn = h + ssm_out
    println("\nAfter residual connection (h + ssm_out):")
    println("  norm: ", round(sqrt(sum(abs2, h_after_attn)), digits=3))
    
    # Post-attention norm
    h_post_norm = layer.post_norm(h_after_attn)
    println("\nAfter post_norm:")
    println("  norm: ", round(sqrt(sum(abs2, h_post_norm)), digits=3))
    
    # MLP
    mlp_out = layer.mlp(h_post_norm)
    println("\nAfter MLP:")
    println("  output norm: ", round(sqrt(sum(abs2, mlp_out)), digits=3))
    
    # Final residual
    h_final = h_after_attn + mlp_out
    println("\nAfter final residual:")
    println("  norm: ", round(sqrt(sum(abs2, h_final)), digits=3))
end

main()
