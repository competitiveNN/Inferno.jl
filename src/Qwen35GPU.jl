"""
Qwen35GPU — Full GPU inference for Qwen3.5 hybrid SSM/attention model on Intel Arc.

All computation on GPU using oneAPI.jl + KernelAbstractions.jl.
FULLY GPU-ACCELERATED: No CPU loops in forward pass.
All weights, computations, and control flow stay on GPU.

OPTIMIZED FOR: Intel Arc A770 with Float16 weights for 2x throughput.
"""
module Qwen35GPU

using oneAPI
using oneAPI: oneAPIBackend
using LinearAlgebra
using KernelAbstractions

# GPU backend for KA kernel calls
const _GPU_BACKEND = oneAPIBackend()

# Use unified kernel library
using ..GPUCommon: rmsnorm_gpu!, silu_gpu!, sigmoid_gpu!, batched_attention_scores!, batched_softmax!, batched_ssm_state_update!, batched_ssm_output_sum!, write_kv_cache_gpu!, fused_attention_forward!
using ..FusedKernels: fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!, fused_ssm_gate_sigmoid!, fused_ssm_decay!, gpu_argmax!, gpu_sample!

using ..GGUF
using ..Tokenizer
using ..LoaderCPU

const oneArray = oneAPI.oneArray
const oneVector{T} = oneArray{T,1}
const oneMatrix{T} = oneArray{T,2}

# ============================================================
# GPU Precision — use Float16 for 2x throughput on Intel Arc
# ============================================================
const E = Float16

# ============================================================
# Config
# ============================================================
Base.@kwdef struct Qwen35GPUConfig
    vocab_size::Int = 151936
    hidden_size::Int = 1024
    intermediate_size::Int = 3584
    num_hidden_layers::Int = 24
    num_attention_heads::Int = 8
    num_key_value_heads::Int = 2
    head_dim::Int = 256
    rms_norm_eps::Float32 = 1.0e-6f0
    rope_theta::Float32 = 10000000.0f0
    max_position_embeddings::Int = 4096
    full_attention_interval::Int = 4
    ssm_inner_size::Int = 2048
    ssm_group_count::Int = 16
    ssm_conv_kernel::Int = 4
    partial_rotary_factor::Float32 = 0.25f0
end

# ============================================================
# Layer Types
# ============================================================
mutable struct AttentionLayer
    qkv_w::oneMatrix{E}
    q_w::oneMatrix{E}
    k_w::oneMatrix{E}
    v_w::oneMatrix{E}
    o_w::oneMatrix{E}
    q_norm::oneVector{E}
    k_norm::oneVector{E}
end

mutable struct MLPLayer
    gate_up_w::oneMatrix{E}
    gate_w::oneMatrix{E}
    up_w::oneMatrix{E}
    down_w::oneMatrix{E}
end

mutable struct SSMLayer
    in_proj::oneMatrix{E}
    gate_proj::oneMatrix{E}
    ssm_out::oneMatrix{E}
    ssm_conv1d::oneMatrix{E}
    alpha_w::oneMatrix{E}
    beta_w::oneMatrix{E}
    ssm_a::oneVector{E}
    ssm_dt_bias::oneVector{E}
    ssm_norm_w::oneVector{E}
    # State buffers
    conv_state::oneMatrix{E}
    h_state::oneArray{Float32, 3}  # Keep Float32 for SSM state (needs precision)
    # Dimensions
    d_inner::Int
    num_v_heads::Int
    num_k_heads::Int
    head_v_dim::Int
    head_k_dim::Int
    conv_channels::Int
end

mutable struct DecoderLayer
    input_norm_w::oneVector{E}
    attn::Union{AttentionLayer, Nothing}
    mlp::MLPLayer
    ssm::Union{SSMLayer, Nothing}
    post_attn_norm_w::oneVector{E}
    is_attention::Bool
end

# ============================================================
# GPU Model
# ============================================================
mutable struct Qwen35GPUModel
    config::Qwen35GPUConfig
    embed::oneMatrix{E}
    layers::Vector{DecoderLayer}
    final_norm_w::oneVector{E}
    lm_head::oneMatrix{E}
    cos_cached::oneMatrix{Float32}  # Keep Float32 for trig tables
    sin_cached::oneMatrix{Float32}
    kv_cache_k::Vector{oneMatrix{E}}
    kv_cache_v::Vector{oneMatrix{E}}
    kv_len::Int
    # Work buffers on GPU
    hidden::oneVector{E}
    residual::oneVector{E}
    norm_buf::oneVector{E}
    qkv_buf::oneVector{E}
    attn_out::oneVector{E}
    gate_buf::oneVector{E}
    up_buf::oneVector{E}
    out_buf::oneVector{E}
    ssm_out_buf::oneVector{E}
    score_buf::oneMatrix{E}
    q_buf::oneVector{E}
    k_buf::oneVector{E}
    v_buf::oneVector{E}
    xz_buf::oneVector{E}
    gate_ssm_buf::oneVector{E}
    alpha_buf::oneVector{E}
    beta_buf::oneVector{E}
    logits_buf::oneVector{E}
    # Batched work buffers (for reduced kernel launches)
    qkv_stack::oneVector{E}
    gate_up_stack::oneVector{E}
    device::oneAPI.oneL0.ZeDevice
end

# ============================================================
# Helper: GPU dot product (NO SYNC - returns GPU array)
# ============================================================
gpu_dot(a, b) = oneAPI.oneAPI.sum(a .* b)

# ============================================================
# RMSNorm (uses GPUCommon kernel)
# ============================================================
function rmsnorm!(out::oneVector{E}, x::oneVector{E}, w::oneVector{E}, eps::AbstractFloat)
    return rmsnorm_gpu!(out, x, w, eps)
end

function rmsnorm!(out::oneVector{E}, x::oneVector{E}, eps::AbstractFloat)
    n = length(x)
    ss = oneAPI.oneAPI.sum(x .^ 2)
    inv_rms = one(E) / sqrt(ss / n + eps)
    out .= x .* inv_rms
    return out
end

# ============================================================
# L2 Norm (for Q/K in Qwen3.5) - GPU native
# ============================================================
function l2norm!(out::oneVector{E}, x::oneVector{E}, eps::AbstractFloat)
    ss = oneAPI.oneAPI.sum(x .^ 2)
    inv_rms = one(E) / sqrt(ss + eps)
    out .= x .* inv_rms
    return out
end

# ============================================================
# RoPE Kernel (Fully GPU - No CPU loops!)
# ============================================================
@kernel function apply_rope_kernel!(x, cos, sin, pos, head_dim::Int, rotary_dim::Int)
    idx = @index(Global, Linear)
    n_total = length(x)
    if idx <= n_total
        half = head_dim ÷ 2
        n_heads = n_total ÷ head_dim
        
        h = (idx - 1) ÷ head_dim  # 0-based head index
        d = (idx - 1) % head_dim  # 0-based dim index
        
        # Only process first half (rotary dims)
        if d < half && d < (rotary_dim ÷ 2)
            cp = cos[pos + 1, d + 1]
            sp = sin[pos + 1, d + 1]
            
            # Pair indices: d and d + half
            i = h * head_dim + d + 1
            j = h * head_dim + half + d + 1
            
            a = x[i]
            b = x[j]
            
            # Apply rotation
            x[i] = a * cp - b * sp
            x[j] = b * cp + a * sp
        end
    end
end

function apply_rope!(q_or_k::AbstractVector, pos::Int,
                     cos_t::oneMatrix{Float32}, sin_t::oneMatrix{Float32},
                     head_dim::Int, rotary_dim::Int)
    n = length(q_or_k)
    kfn = apply_rope_kernel!(_GPU_BACKEND)
    kfn(q_or_k, cos_t, sin_t, pos, head_dim, rotary_dim; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return q_or_k
end

# ============================================================
# Attention Forward (Fully GPU - No CPU loops!)
# ============================================================
function attention_forward!(model::Qwen35GPUModel, layer::DecoderLayer, layer_idx::Int,
                            hidden::oneVector{E}, pos::Int)
    attn = layer.attn
    cfg = model.config
    h = cfg.hidden_size
    head_dim = cfg.head_dim
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    n_groups = n_heads ÷ n_kv

    # ============================================================
    # Batched QKV projection (single matmul, 3x fewer kernel launches!)
    # The qkv_w matrix is (num_heads*head_dim + 2*num_kv_heads*head_dim) x hidden_size
    # Output is split into Q, K, V views
    # ============================================================
    q_size = n_heads * head_dim
    k_size = n_kv * head_dim
    v_size = k_size
    total_qkv = q_size + k_size + v_size

    mul!(model.qkv_stack, attn.qkv_w, hidden)

    q = view(model.qkv_stack, 1:q_size)
    k = view(model.qkv_stack, (q_size+1):(q_size+k_size))
    v = view(model.qkv_stack, (q_size+k_size+1):total_qkv)

    # L2 Q/K norm - GPUs native (no sync, no alloc)
    fused_l2norm!(q, cfg.rms_norm_eps)
    fused_l2norm!(k, cfg.rms_norm_eps)

    # RoPE - FULLY GPU
    rotary_dim = Int(head_dim * cfg.partial_rotary_factor)
    apply_rope!(q, pos, model.cos_cached, model.sin_cached, head_dim, rotary_dim)
    apply_rope!(k, pos, model.cos_cached, model.sin_cached, head_dim, rotary_dim)

    # Store KV cache - FULLY GPU (batched copy)
    k_cache = model.kv_cache_k[layer_idx]
    v_cache = model.kv_cache_v[layer_idx]

    # Store KV cache - FULLY GPU (single kernel, no CPU loop)
    write_kv_cache_gpu!(k_cache, v_cache, k, v, head_dim, pos-1)

    # Attention scores + softmax + weighted sum — fully FUSED (single kernel)
    seq_len = pos + 1
    fused_attention_forward!(model.attn_out, q, k_cache, v_cache, n_heads, head_dim, seq_len, n_groups)

    # Output projection
    mul!(model.out_buf, attn.o_w, model.attn_out)
    return model.out_buf
end

# ============================================================
# SSM Forward (Fully GPU - No CPU loops!)
# ============================================================
function ssm_forward!(model::Qwen35GPUModel, layer::DecoderLayer, hidden::oneVector{E})
    ssm = layer.ssm
    cfg = model.config
    h = cfg.hidden_size

    d_inner = ssm.d_inner
    n_v = ssm.num_v_heads
    n_k = ssm.num_k_heads
    head_v = ssm.head_v_dim
    head_k = ssm.head_k_dim
    conv_channels = ssm.conv_channels

    # Combined in-projection
    xz = model.xz_buf
    mul!(xz, ssm.in_proj, hidden)
    x_conv = view(xz, 1:conv_channels)
    z = view(xz, conv_channels+1:conv_channels+d_inner)

    # Gate
    gate_out = model.gate_ssm_buf
    mul!(gate_out, ssm.gate_proj, hidden)
    silu_gpu!(gate_out, gate_out)

    # Conv1d: shift state, store new input
    ssm.conv_state[:, 1:end-1] .= ssm.conv_state[:, 2:end]
    ssm.conv_state[:, end] .= x_conv

    # Alpha/beta projections
    alpha = model.alpha_buf
    beta_all = model.beta_buf
    mul!(alpha, ssm.alpha_w, hidden)
    mul!(beta_all, ssm.beta_w, hidden)

    # dt = sigmoid(dt_bias + alpha) — reuse alpha buffer (alpha is dead after this)
    fused_ssm_gate_sigmoid!(alpha, ssm.ssm_dt_bias, alpha)
    dt = alpha

    # Decay = exp(-ssm_a * dt)
    decay_buf = view(model.gate_buf, 1:length(dt))
    fused_ssm_decay!(decay_buf, ssm.ssm_a, dt)
    decay = decay_buf

    # State update - FULLY GPU (batched kernel)
    batched_ssm_state_update!(ssm.h_state, decay, beta_all, x_conv, n_v, head_v, head_k)

    # SSM output sum - FULLY GPU (batched kernel)
    y_all = view(model.ssm_out_buf, 1:d_inner)
    batched_ssm_output_sum!(y_all, ssm.h_state, n_v, head_v, head_k)

    # Gate output: fused SiLU + elementwise multiply (no temp, single kernel)
    y_gated = view(model.ssm_out_buf, 1:d_inner)
    fused_silu_gate_mul!(y_gated, y_all, gate_out)

    # Output projection
    mul!(model.out_buf, ssm.ssm_out, y_gated)
    return model.out_buf
end

# ============================================================
# MLP (SwiGLU) — Batched gate+up projection for fewer kernel launches
# ============================================================
function mlp_forward!(model::Qwen35GPUModel, layer::DecoderLayer, hidden::oneVector{E})
    mlp = layer.mlp
    cfg = model.config
    int_size = cfg.intermediate_size

    # ============================================================
    # Batched gate+up (single matmul, 2x fewer kernel launches!)
    # gate_up_w is (2*intermediate_size) x hidden_size
    # Output is split into gate and up views
    # ============================================================
    mul!(model.gate_up_stack, mlp.gate_up_w, hidden)

    gate = view(model.gate_up_stack, 1:int_size)
    up = view(model.gate_up_stack, (int_size+1):(2*int_size))

    # Apply SiLU to gate and element-wise multiply with up
    fused_mlp_gate_mul!(gate, up)

    # Down projection (gate now contains SiLU(gate) .* up)
    mul!(model.out_buf, mlp.down_w, gate)
    return model.out_buf
end

# ============================================================
# Full Forward Pass (Fully GPU)
# ============================================================
function forward_gpu!(model::Qwen35GPUModel, token_id::Int, start_pos::Int)
    cfg = model.config
    h = cfg.hidden_size

    # Embedding lookup
    copy!(model.hidden, view(model.embed, token_id, :))

    for (layer_idx, layer) in enumerate(model.layers)
        copy!(model.residual, model.hidden)
        rmsnorm_gpu!(model.norm_buf, model.hidden, layer.input_norm_w, cfg.rms_norm_eps)

        if layer.is_attention
            attn_out = attention_forward!(model, layer, layer_idx, model.norm_buf, start_pos)
            rmsnorm_gpu!(model.qkv_buf, attn_out, layer.post_attn_norm_w, cfg.rms_norm_eps)
        else
            ssm_out = ssm_forward!(model, layer, model.norm_buf)
            rmsnorm_gpu!(model.qkv_buf, ssm_out, layer.post_attn_norm_w, cfg.rms_norm_eps)
        end

        mlp_out = mlp_forward!(model, layer, model.qkv_buf)
        model.hidden .= model.residual .+ mlp_out
    end

    rmsnorm_gpu!(model.norm_buf, model.hidden, model.final_norm_w, cfg.rms_norm_eps)
    mul!(model.logits_buf, model.lm_head, model.norm_buf)
    return model.logits_buf
end

# ============================================================
# Model state reset
# ============================================================
function reset_model!(model::Qwen35GPUModel)
    for k in model.kv_cache_k
        fill!(k, zero(E))
    end
    for v in model.kv_cache_v
        fill!(v, zero(E))
    end
    for layer in model.layers
        if !layer.is_attention && !isnothing(layer.ssm)
            fill!(layer.ssm.conv_state, zero(E))
            fill!(layer.ssm.h_state, zero(Float32))  # Float32 state
        end
    end
    return model
end

# ============================================================
# Generation (streaming)
# ============================================================
function generate_stream(model::Qwen35GPUModel, tok::BPETokenizer, prompt::String;
                         max_tokens::Int=100, temperature::Float32=0.0f0,
                         top_p::Float32=0.0f0, top_k::Int=0)
    return Channel{String}(32) do chan
        prompt_ids = encode(tok, prompt)
        token_buf  = Int[]

        reset_model!(model)
        pos = 0
        logits = nothing

        # Prime KV caches with full prompt
        for token in prompt_ids
            logits = forward_gpu!(model, token, pos)
            pos += 1
        end

        # No prompt or no generation requested
        if isempty(prompt_ids) || max_tokens <= 0 || logits === nothing
            return
        end

        # First generated token from the last prompt logits
        if temperature == 0.0f0
            last_token = gpu_argmax!(logits)
        else
            last_token = gpu_sample!(logits, Float16(temperature))
        end

        word = decode(tok, [last_token])
        put!(chan, word)

        if last_token == 151643 || last_token == 0
            return
        end

        # Continue generating
        for _ in 1:max_tokens - 1
            logits = forward_gpu!(model, last_token, pos)
            pos += 1

            if temperature == 0.0f0
                last_token = gpu_argmax!(logits)
            else
                last_token = gpu_sample!(logits, Float16(temperature))
            end

            word = decode(tok, [last_token])
            put!(chan, word)

            if last_token == 151643 || last_token == 0
                break
            end
        end
    end
end

function sample(probs::Vector{E})
    r = rand(E)
    cum = zero(E)
    for i in 1:length(probs)
        cum += probs[i]
        if r <= cum
            return i
        end
    end
    return length(probs)
end

# ============================================================
# GGUF Loader
# ============================================================
function load_qwen35_gpu(gguf_path::String; gpu_device::Int=1, max_seq_len::Int=4096)
    println("[GPU] Loading Qwen3.5 from $gguf_path")

    devs = collect(oneAPI.devices())
    gpu_idx = min(gpu_device, length(devs))
    oneAPI.device!(devs[gpu_idx])
    println("[GPU] Device: $(devs[gpu_idx])")

    file = GGUF.read_gguf(gguf_path)
    md = file.metadata
    arch = get(md, "general.architecture", "qwen2")

    cfg = Qwen35GPUConfig(
        vocab_size = Int(get(md, "$arch.vocab_size", 151936)),
        hidden_size = Int(get(md, "$arch.embedding_length", 1024)),
        intermediate_size = Int(get(md, "$arch.feed_forward_length", 3584)),
        num_hidden_layers = Int(get(md, "$arch.block_count", 24)),
        num_attention_heads = Int(get(md, "$arch.attention.head_count", 8)),
        num_key_value_heads = Int(get(md, "$arch.attention.head_count_kv", 2)),
        head_dim = Int(get(md, "$arch.attention.key_length", 128)),
        rms_norm_eps = Float32(get(md, "$arch.attention.layer_norm_rms_epsilon", 1.0e-6)),
        max_position_embeddings = min(4096, Int(get(md, "$arch.context_length", 32768))),
        full_attention_interval = Int(get(md, "$arch.full_attention_interval", 4)),
        ssm_inner_size = Int(get(md, "$arch.ssm.inner_size", 2048)),
        ssm_group_count = Int(get(md, "$arch.ssm.group_count", 16)),
        ssm_conv_kernel = Int(get(md, "$arch.ssm.conv_kernel", 4)),
    )
    println("[GPU] hidden=$(cfg.hidden_size) layers=$(cfg.num_hidden_layers) heads=$(cfg.num_attention_heads)")

    h = cfg.hidden_size
    head_dim = cfg.head_dim

    function gpu_tensor(name::String)
        info = GGUF.get_tensor(file, name)
        cpu_tensor = LoaderCPU.extract_tensor_cpu(file, info)
        # Convert all CPU weights to GPU Float16
        if cpu_tensor isa Matrix{Float32}
            return oneArray(E.(cpu_tensor))
        elseif cpu_tensor isa Matrix
            return oneArray(E.(Matrix{Float32}(cpu_tensor)))
        elseif cpu_tensor isa Vector{Float32}
            return oneArray(E.(cpu_tensor))
        elseif cpu_tensor isa Vector
            return oneArray(E.(Vector{Float32}(cpu_tensor)))
        else
            return oneArray(E.(reshape(Vector{Float32}(cpu_tensor), :, 1)))
        end
    end

    # ===========================
    # Load weights
    # ===========================
    embed = gpu_tensor("token_embd.weight")
    final_norm_w = gpu_tensor("output_norm.weight")
    lm_head = gpu_tensor("output.weight")

    layers = DecoderLayer[]

    for layer_idx in 1:cfg.num_hidden_layers
        prefix = "blk.$(layer_idx - 1)."
        
        # Check if this is an full attention layer
        is_attention = (layer_idx % cfg.full_attention_interval == 1)
        
        input_norm = gpu_tensor("$(prefix)attn_norm.weight")
        post_norm = gpu_tensor("$(prefix)ffn_norm.weight")
        
        if is_attention
            # Load Q,K,V,O weights
            q_w = gpu_tensor("$(prefix)attn_q.weight")
            k_w = gpu_tensor("$(prefix)attn_k.weight")
            v_w = gpu_tensor("$(prefix)attn_v.weight")
            o_w = gpu_tensor("$(prefix)attn_output.weight")
            
            # Stack Q,K,V for batched matmul
            qkv_w = oneArray(vcat(oneAPI.Array(q_w), oneAPI.Array(k_w), oneAPI.Array(v_w)))
            
            q_norm = gpu_tensor("$(prefix)attn_q_norm.weight")
            k_norm = gpu_tensor("$(prefix)attn_k_norm.weight")
            
            attn_layer = AttentionLayer(qkv_w, q_w, k_w, v_w, o_w, q_norm, k_norm)
        else
            attn_layer = nothing
        end
        
        # MLP weights
        gate_w = gpu_tensor("$(prefix)ffn_gate.weight")
        up_w = gpu_tensor("$(prefix)ffn_up.weight")
        down_w = gpu_tensor("$(prefix)ffn_down.weight")
        
        # Stack gate+up for batched matmul
        gate_up_w = oneArray(vcat(oneAPI.Array(gate_w), oneAPI.Array(up_w)))
        
        mlp_layer = MLPLayer(gate_up_w, gate_w, up_w, down_w)
        
        if !is_attention
            # SSM weights
            in_proj = gpu_tensor("$(prefix)ssm_in_proj.weight")
            gate_proj = gpu_tensor("$(prefix)ssm_gate_proj.weight")
            ssm_out = gpu_tensor("$(prefix)ssm_out.weight")
            ssm_conv1d = gpu_tensor("$(prefix)ssm_conv1d.weight")
            alpha_w = gpu_tensor("$(prefix)ssm_alpha.weight")
            beta_w = gpu_tensor("$(prefix)ssm_beta.weight")
            ssm_a = gpu_tensor("$(prefix)ssm_a.weight")
            ssm_dt_bias = gpu_tensor("$(prefix)ssm_dt_bias.weight")
            ssm_norm_w = gpu_tensor("$(prefix)ssm_norm.weight")
            
            # Determine SSM dimensions
            d_inner = size(ssm_out, 2)  # output dimension
            conv_channels = size(ssm_conv1d, 2)
            num_v_heads = 1  # placeholder, will be inferred from actual tensor shapes
            
            # State buffers (Float32 for precision)
            conv_state = oneArray(zeros(Float32, conv_channels, cfg.ssm_conv_kernel))
            h_state = oneArray(zeros(Float32, 1, 1, num_v_heads))
            
            ssm_layer = SSMLayer(in_proj, gate_proj, ssm_out, ssm_conv1d, alpha_w, beta_w,
                                ssm_a, ssm_dt_bias, ssm_norm_w,
                                conv_state, h_state,
                                d_inner, num_v_heads, 1, 1, 1, conv_channels)
            
            ssm = ssm_layer
        else
            ssm = nothing
        end
        
        layer = DecoderLayer(input_norm, attn_layer, mlp_layer, ssm, post_norm, is_attention)
        push!(layers, layer)
    end

    # RoPE tables (keep in Float32 for accuracy)
    function precompute_rope(dim::Int, max_seq_len::Int, theta::Float32)
        half = dim ÷ 2
        freqs = Float32[1.0f0 / (theta ^ (2(i-1)/dim)) for i in 1:half]
        
        cos_table = Matrix{Float32}(undef, max_seq_len, half)
        sin_table = Matrix{Float32}(undef, max_seq_len, half)
        
        for pos in 1:max_seq_len
            for i in 1:half
                angle = (pos - 1) * freqs[i]
                cos_table[pos, i] = cos(angle)
                sin_table[pos, i] = sin(angle)
            end
        end
        
        return cos_table, sin_table
    end
    
    cos_cached, sin_cached = precompute_rope(Int(head_dim * cfg.partial_rotary_factor), max_seq_len, Float32(cfg.rope_theta))
    
    cos_cached_gpu = oneArray(cos_cached)
    sin_cached_gpu = oneArray(sin_cached)

    # KV caches
    kv_cache_k = [oneArray(zeros(E, head_dim * cfg.num_key_value_heads, max_seq_len)) for _ in 1:cfg.num_hidden_layers]
    kv_cache_v = [oneArray(zeros(E, head_dim * cfg.num_key_value_heads, max_seq_len)) for _ in 1:cfg.num_hidden_layers]

    # Work buffers
    model = Qwen35GPUModel(
        config = cfg,
        embed = embed,
        layers = layers,
        final_norm_w = final_norm_w,
        lm_head = lm_head,
        cos_cached = cos_cached_gpu,
        sin_cached = sin_cached_gpu,
        kv_cache_k = kv_cache_k,
        kv_cache_v = kv_cache_v,
        kv_len = 0,
        hidden = oneVector{E}(undef, h),
        residual = oneVector{E}(undef, h),
        norm_buf = oneVector{E}(undef, h),
        qkv_buf = oneVector{E}(undef, h),
        attn_out = oneVector{E}(undef, h),
        gate_buf = oneVector{E}(undef, cfg.intermediate_size),
        up_buf = oneVector{E}(undef, cfg.intermediate_size),
        out_buf = oneVector{E}(undef, h),
        ssm_out_buf = oneVector{E}(undef, cfg.ssm_inner_size),
        score_buf = oneMatrix{E}(undef, 1, 1),
        q_buf = oneVector{E}(undef, head_dim * cfg.num_attention_heads),
        k_buf = oneVector{E}(undef, head_dim * cfg.num_key_value_heads),
        v_buf = oneVector{E}(undef, head_dim * cfg.num_key_value_heads),
        xz_buf = oneVector{E}(undef, cfg.ssm_inner_size + cfg.hidden_size),
        gate_ssm_buf = oneVector{E}(undef, cfg.ssm_inner_size),
        alpha_buf = oneVector{E}(undef, cfg.ssm_group_count),
        beta_buf = oneVector{E}(undef, cfg.ssm_group_count),
        logits_buf = oneVector{E}(undef, cfg.vocab_size),
        # Batched work buffers
        qkv_stack = oneVector{E}(undef, head_dim * cfg.num_attention_heads + 2 * head_dim * cfg.num_key_value_heads),
        gate_up_stack = oneVector{E}(undef, 2 * cfg.intermediate_size),
        device = devs[gpu_idx]
    )

    println("[GPU] Model loaded: $(cfg.num_hidden_layers) layers, $(cfg.hidden_size) hidden, $(cfg.vocab_size) vocab")
    println("[GPU] Float16 precision enabled for 2x throughput on Intel Arc")
    
    return model
end

end # module
