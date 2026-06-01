using Inferno
using Statistics

function main()
    model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    tokens = file.metadata["tokenizer.ggml.tokens"]

    # Check an attention layer
    println("=== Checking Attention Layer (layer 3) ===")
    attn = model.layers[4].op  # Layer 3 is attention (0-indexed)
    println("n_heads: ", attn.n_heads)
    println("n_kv: ", attn.n_kv)
    println("head_dim: ", attn.head_dim)
    println("wq size: ", size(attn.wq))
    println("wk size: ", size(attn.wk))
    println("wv size: ", size(attn.wv))
    println("wo size: ", size(attn.wo))

    # Test with a single token
    caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
    ModelCPU.reset_states_cpu!(model)

    # Get embedding for token 561 (" The")
    x = copy(view(model.embed, :, 561))
    println("\n=== Processing token 561 (\" The\") ===")
    println("Embedding stats: mean=", mean(x), " std=", std(x))

    # Process through first few layers
    for (i, layer) in enumerate(model.layers[1:4])
        x_before = copy(x)
        x = layer(x, 0, model.rope, caches[i])
        println("After layer $i ($(layer.is_ssm ? "SSM" : "Attn")): mean=$(mean(x)), std=$(std(x)), any NaN=$(any(isnan, x)))")
    end
end

main()
