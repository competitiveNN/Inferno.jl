module Gemma4Loader

using JSON3
using LinearAlgebra
using ..Gemma4
using ..Gemma4:
    Gemma4Model, Gemma4Config, DecoderLayer, AttentionLayer, MLPLayer,
    PerLayerInput, KVCacheG4, init_kv_cache, precompute_rope, compute_store_kv_layers
using ..Tokenizer: BPETokenizer, encode, decode
using ..Safetensors: parse_safetensors, get_tensor, SafetensorsFile, load_hf_tokenizer
using ..LoaderCPU: load_safetensors_model

function load_gemma4(model_dir::String; max_seq_len::Int=2048)
    println("Loading Gemma4 model from $model_dir")

    # Load config
    config_path = joinpath(model_dir, "config.json")
    config_raw = JSON3.read(read(config_path, String))
    tc = config_raw.text_config

    # Parse layer types
    layer_types = String[]
    for lt in tc.layer_types
        push!(layer_types, String(lt))
    end

    # KV sharing
    num_layers = Int(tc.num_hidden_layers)
    num_kv_shared = get(tc, :num_kv_shared_layers, 0)
    first_kv_shared = num_layers - num_kv_shared  # 0-based index of first shared layer

    # Determine per-layer intermediate sizes
    intermediate_sizes = Int[]
    if haskey(tc, :per_layer_intermediate_sizes)
        for s in tc.per_layer_intermediate_sizes
            push!(intermediate_sizes, Int(s))
        end
    else
        base_inter = Int(tc.intermediate_size)
        use_double_wide = get(tc, :use_double_wide_mlp, false)
        for i in 0:(num_layers-1)
            is_shared = i >= first_kv_shared && num_kv_shared > 0
            if use_double_wide && is_shared
                push!(intermediate_sizes, base_inter * 2)
            else
                push!(intermediate_sizes, base_inter)
            end
        end
    end

    # Attention config
    attention_k_eq_v = get(tc, :attention_k_eq_v, false)
    raw_global_kv = get(tc, :num_global_key_value_heads, nothing)
    num_global_kv_heads = raw_global_kv === nothing ? Int(tc.num_key_value_heads) : Int(raw_global_kv)

    # Per-layer input config
    pli_size = get(tc, :hidden_size_per_layer_input, 0)
    vocab_per_layer = get(tc, :vocab_size_per_layer_input, 0)

    # RoPE params - extract from rope_parameters
    sliding_rope_theta = 10000.0f0
    full_rope_theta = 1000000.0f0
    full_partial_rotary_factor = 0.25f0
    if haskey(tc, :rope_parameters)
        for (lt, params) in pairs(tc.rope_parameters)
            if String(lt) == "sliding_attention" && params !== nothing
                if haskey(params, :rope_theta)
                    sliding_rope_theta = Float32(params.rope_theta)
                end
            elseif String(lt) == "full_attention" && params !== nothing
                if haskey(params, :rope_theta)
                    full_rope_theta = Float32(params.rope_theta)
                end
                if haskey(params, :partial_rotary_factor)
                    full_partial_rotary_factor = Float32(params.partial_rotary_factor)
                end
            end
        end
    end

    config = Gemma4Config(
        Int(tc.hidden_size),
        num_layers,
        Int(tc.num_attention_heads),
        Int(tc.num_key_value_heads),
        num_global_kv_heads,
        Int(get(tc, :head_dim, tc.hidden_size ÷ tc.num_attention_heads)),
        Int(get(tc, :global_head_dim, 0)),
        Int(tc.intermediate_size),
        maximum(intermediate_sizes), # double_wide_intermediate
        Int(tc.vocab_size),
        vocab_per_layer,
        max_seq_len,
        Int(get(tc, :sliding_window, 512)),
        Float32(get(tc, :rms_norm_eps, 1e-6)),
        Float32(get(tc, :final_logit_softcapping, 0.0)),
        Float32(get(tc, :attention_logits_soft_cap, 0.0)),
        Float32(sqrt(Float64(tc.hidden_size))), # embed_scale
        Float32(2.0^-0.5), # per_layer_input_scale = 1/sqrt(2)
        Float32(Float64(tc.hidden_size)^-0.5), # per_layer_model_projection_scale
        layer_types,
        num_kv_shared,
        first_kv_shared,
        pli_size,
        attention_k_eq_v,
        # RoPE params
        sliding_rope_theta,
        full_rope_theta,
        full_partial_rotary_factor,
        Bool(get(config_raw, :tie_word_embeddings, true)),
        Int[], # store_kv_layers — will be computed below
    )

    # Compute store_kv_layers
    store_kv_layers = compute_store_kv_layers(config)
    config = Gemma4Config(
        config.hidden_size, config.num_layers,
        config.num_q_heads, config.num_kv_heads, config.num_global_kv_heads,
        config.head_dim, config.global_head_dim, config.intermediate_size,
        config.double_wide_intermediate, config.vocab_size, config.vocab_size_per_layer_input,
        config.max_seq_len, config.sliding_window, config.rms_norm_eps,
        config.final_logit_softcapping, config.attention_logits_softcapping,
        config.embed_scale, config.per_layer_input_scale, config.per_layer_model_projection_scale,
        config.layer_types, config.num_kv_shared_layers, config.first_kv_shared_layer,
        config.hidden_size_per_layer_input, config.attention_k_eq_v,
        config.sliding_rope_theta, config.full_rope_theta, config.full_partial_rotary_factor,
        config.tie_word_embeddings, store_kv_layers
    )

    println("  Config: $(config.hidden_size) hidden, $(config.num_layers) layers, $(config.head_dim) head_dim")
    println("  Sliding window: $(config.sliding_window), KV shared: $(config.num_kv_shared_layers)")
    println("  Per-layer input: $(config.hidden_size_per_layer_input)")
    println("  Store KV layers: $(config.store_kv_layers)")

    # Find safetensors files
    safetensors_files = filter(f -> endswith(f, ".safetensors"), readdir(model_dir))
    if isempty(safetensors_files)
        error("No safetensors files found in $model_dir")
    end

    # Load safetensors
    prefix = "model.language_model."
    st = parse_safetensors(joinpath(model_dir, safetensors_files[1]))

    # Load embed_tokens
    embed_tokens = get_tensor(st, "$(prefix)embed_tokens.weight")
    println("  Embed tokens: $(size(embed_tokens))")

    # Load per-layer input embeddings
    embed_per_layer = Matrix{Float32}(undef, 0, 0)
    per_layer_model_proj = Matrix{Float32}(undef, 0, 0)
    per_layer_proj_norm_w = Vector{Float32}(undef, 0)
    if pli_size > 0
        embed_per_layer = get_tensor(st, "$(prefix)embed_tokens_per_layer.weight")
        per_layer_model_proj = get_tensor(st, "$(prefix)per_layer_model_projection.weight")
        # Scale projection weights at load time (HF does this in forward)
        per_layer_model_proj .*= config.per_layer_model_projection_scale
        per_layer_proj_norm_w = Float32.(vec(get_tensor(st, "$(prefix)per_layer_projection_norm.weight")))
        println("  Per-layer embed: $(size(embed_per_layer)), proj: $(size(per_layer_model_proj))")
    end

    # Load final norm
    final_norm_w = Float32.(vec(get_tensor(st, "$(prefix)norm.weight")))

    # Load layers
    layers = DecoderLayer[]
    for i in 0:(num_layers-1)
        layer_type = layer_types[i+1]
        is_sliding = layer_type == "sliding_attention"
        is_kv_shared = i >= first_kv_shared && num_kv_shared > 0

        # Head dim for this layer
        if !is_sliding && config.global_head_dim > 0
            layer_head_dim = config.global_head_dim
            layer_num_kv = config.num_global_kv_heads
        else
            layer_head_dim = config.head_dim
            layer_num_kv = config.num_kv_heads
        end

        inter_size = intermediate_sizes[i+1]

        # Find KV shared source
        kv_shared_src = -1
        if is_kv_shared
            # Find the last non-shared layer of the same type (0-based)
            for j in (first_kv_shared-1):-1:0
                if layer_types[j+1] == layer_type
                    kv_shared_src = j
                    break
                end
            end
        end

        # Attention weights
        q_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.self_attn.q_proj.weight"))
        q_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.self_attn.q_norm.weight")))

        if !is_kv_shared
            k_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.self_attn.k_proj.weight"))
            k_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.self_attn.k_norm.weight")))
        else
            k_proj = Matrix{Float32}(undef, 0, 0)
            k_norm_w = Vector{Float32}(undef, 0)
        end

        # V proj: may not exist if k_eq_v or shared KV layer
        v_proj_key = "$(prefix)layers.$i.self_attn.v_proj.weight"
        v_proj = Matrix{Float32}(undef, 0, 0)
        try
            v_proj = Float32.(get_tensor(st, v_proj_key))
        catch
            # k_eq_v mode or shared KV layer — no v_proj
        end

        o_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.self_attn.o_proj.weight"))

        # Layer norms
        input_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.input_layernorm.weight")))
        post_attn_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.post_attention_layernorm.weight")))
        pre_ff_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.pre_feedforward_layernorm.weight")))
        post_ff_norm_w = Float32.(vec(get_tensor(st, "$(prefix)layers.$i.post_feedforward_layernorm.weight")))

        # MLP
        gate_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.mlp.gate_proj.weight"))
        up_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.mlp.up_proj.weight"))
        down_proj = Float32.(get_tensor(st, "$(prefix)layers.$i.mlp.down_proj.weight"))

        # Per-layer input
        pli = nothing
        if pli_size > 0
            pli_gate_key = "$(prefix)layers.$i.per_layer_input_gate.weight"
            pli_proj_key = "$(prefix)layers.$i.per_layer_projection.weight"
            pli_post_norm_key = "$(prefix)layers.$i.post_per_layer_input_norm.weight"
            try
                pli_gate = Float32.(get_tensor(st, pli_gate_key))
                pli_proj = Float32.(get_tensor(st, pli_proj_key))
                pli_post_norm = Float32.(vec(get_tensor(st, pli_post_norm_key)))
                pli = PerLayerInput(pli_gate, pli_proj, pli_post_norm)
            catch e
                println("  Warning: per-layer input weights not found for layer $i: $e")
            end
        end

        # Layer scalar
        layer_scalar_key = "$(prefix)layers.$i.layer_scalar"
        layer_scalar_val = try
            Float32(vec(get_tensor(st, layer_scalar_key))[1])
        catch
            1.0f0
        end

        n = config.hidden_size

        # Create attention layer with pre-allocated buffers
        q_buf = Vector{Float32}(undef, config.num_q_heads * layer_head_dim)
        k_buf = Vector{Float32}(undef, layer_num_kv * layer_head_dim)
        v_buf = Vector{Float32}(undef, layer_num_kv * layer_head_dim)
        attn_out_buf = Vector{Float32}(undef, config.num_q_heads * layer_head_dim)

        attn = AttentionLayer(
            q_proj, k_proj, v_proj, o_proj,
            q_norm_w, k_norm_w,
            is_sliding, is_kv_shared, kv_shared_src,
            layer_head_dim, layer_num_kv,
            q_buf, k_buf, v_buf, attn_out_buf
        )

        mlp = MLPLayer(
            gate_proj, up_proj, down_proj,
            Vector{Float32}(undef, inter_size), # gate_buf
            Vector{Float32}(undef, inter_size), # up_buf
            Vector{Float32}(undef, n), # hidden_buf
        )

        layer = DecoderLayer(
            input_norm_w, post_attn_norm_w, pre_ff_norm_w, post_ff_norm_w,
            attn, mlp, pli, layer_scalar_val,
            Vector{Float32}(undef, n), # norm_buf
            Vector{Float32}(undef, pli_size), # pli_gate_buf
            Vector{Float32}(undef, n), # pli_out_buf
        )

        push!(layers, layer)
    end

    # Pre-compute RoPE
    sliding_cos, sliding_sin, full_cos, full_sin = precompute_rope(config, max_seq_len)

    # Create model
    model = Gemma4Model(
        config, embed_tokens, embed_per_layer, per_layer_model_proj, per_layer_proj_norm_w,
        final_norm_w, layers,
        sliding_cos, sliding_sin, full_cos, full_sin,
        Dict{Int, Matrix{Float32}}(), Dict{Int, Matrix{Float32}}(),
        Vector{Float32}(undef, config.hidden_size), # hidden_buf
        Vector{Float32}(undef, config.hidden_size), # residual_buf
        Vector{Float32}(undef, num_layers * max(pli_size, 1)), # pli_embed_buf
        Vector{Float32}(undef, num_layers * max(pli_size, 1)), # pli_proj_buf
        Vector{Float32}(undef, config.vocab_size), # logits_buf
        Vector{Float32}(undef, config.hidden_size), # logits_hidden_scaled
    )

    # Load tokenizer
    tok = nothing
    tok_path = joinpath(model_dir, "tokenizer.json")
    if isfile(tok_path)
        tok = load_hf_tokenizer(model_dir)
        println("  Tokenizer loaded: vocab_size=$(length(tok.id_to_token))")
    else
        println("  Warning: No tokenizer.json found")
    end

    println("  Model loaded successfully!")
    return model, tok
end

end # module Gemma4Loader
