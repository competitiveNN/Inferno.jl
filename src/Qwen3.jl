"""
Qwen3 (non-SSM) CPU Inference Implementation

Pure Julia CPU backend for Qwen3 pure transformer architecture.
Supports GGUF and Safetensors format.

# Architecture
- Pure transformer with full attention in ALL layers (NO SSM)
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE) - FULL head_dim (not partial)
- RMSNorm (standard, no +1 convention)
- SwiGLU MLP
- L2 normalization on Q/K

Key differences from Qwen3.5:
1. NO SSM/GatedDeltaNet layers - all layers use full attention
2. FULL rotary embeddings (all head_dim gets RoPE)
3. No Gated Attention (q_proj outputs only queries)
4. No partial rotary factor

# Usage
```julia
model = load_qwen3_cpu("model.gguf")
output = generate_cpu(model, tokenizer, "Hello, "; max_tokens=50)
```
"""
module Qwen3

using LinearAlgebra
using Statistics
using LoopVectorization
using ..CommonOps
using ..QuantsCPU
using ..ArrowLake
using ..QuantizedKernels
using BFloat16s
using Printf
using JSON
using ..GGUF
using ..Dequant
using ..Tokenizer

export Qwen3ConfigCPU, Qwen3ModelCPU, KVCacheCPU, forward_cpu!, RMSNormCPU, MLPCPU, 
       FullAttentionCPU, DecoderLayerCPU, RotaryEmbeddingCPU
export init_kv_cache_cpu, reset_states_cpu!, softmax_sample, generate_cpu, generate_stream_cpu
export load_qwen3_cpu, stream_to_stdout_qwen3
export load_qwen3_safetensors

# --- Configuration ---
Base.@kwdef struct Qwen3ConfigCPU
    architecture::Symbol = :qwen3
    vocab_size::Int = 151936
    hidden_size::Int = 2560
    intermediate_size::Int = 9728
    num_hidden_layers::Int = 36
    num_attention_heads::Int = 32
    num_key_value_heads::Int = 8
    head_dim::Int = 128
    rms_norm_eps::Float32 = 1e-6f0
    rope_theta::Float32 = 1000000.0f0
    max_position_embeddings::Int = 32768
    # Performance options
    use_bf16_weights::Bool = has_arrow_lake_features()
end

# Helper functions
sigmoid(x) = 1.0f0 / (1.0f0 + exp(-x))
silu(x::AbstractArray) = x .* sigmoid.(x)

# --- RMS Norm ---
struct RMSNormCPU
    weight::Vector{Float32}
    eps::Float32
end

function RMSNormCPU(weight::AbstractArray{Float32}, eps::Float32)
    return RMSNormCPU(vec(weight), eps)
end

@inline function rmsnorm_cpu!(out::AbstractArray{Float32}, x::AbstractArray{Float32}, norm::RMSNormCPU)
    ss = zero(Float32)
    @turbo for i in eachindex(x)
        ss += x[i] * x[i]
    end
    scale = 1.0f0 / sqrt(ss / length(x) + norm.eps)
    @turbo for i in eachindex(x)
        out[i] = x[i] * scale * norm.weight[i]
    end
    return out
end

# --- Rotary Position Embedding (Full on all head_dim) ---
struct RotaryEmbeddingCPU
    inv_freq::Vector{Float32}
    max_seq_len::Int
    rotary_dim::Int
    cos_cache::Matrix{Float32}  # (half, max_seq_len)
    sin_cache::Matrix{Float32}  # (half, max_seq_len)
end

function RotaryEmbeddingCPU(head_dim::Int, theta::Float32 = 10000.0f0, max_seq_len::Int = 4096)
    # Full rotary: all head_dim gets RoPE
    half = div(head_dim, 2)
    inv_freq = Float32[1.0 / (theta ^ (2(i-1)/head_dim)) for i in 1:half]
    
    cos_cache = Matrix{Float32}(undef, half, max_seq_len)
    sin_cache = Matrix{Float32}(undef, half, max_seq_len)
    
    for pos in 1:max_seq_len
        for i in 1:half
            freq = inv_freq[i] * (pos - 1)
            cos_cache[i, pos] = cos(freq)
            sin_cache[i, pos] = sin(freq)
        end
    end
    
    return RotaryEmbeddingCPU(inv_freq, max_seq_len, head_dim, cos_cache, sin_cache)
end

function apply_rotary_emb!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rope.rotary_dim, 2)
    
    pos_idx = pos + 1
    @inbounds begin
        cos_vals = view(rope.cos_cache, :, pos_idx)
        sin_vals = view(rope.sin_cache, :, pos_idx)
    end
    
    for h in 1:num_heads
        for i in 1:half
            @inbounds begin
                idx1, idx2 = i, i + half
                x1, x2 = x[idx1, h], x[idx2, h]
                c, s = cos_vals[i], sin_vals[i]
                x[idx1, h] = x1 * c - x2 * s
                x[idx2, h] = x1 * s + x2 * c
            end
        end
    end
    return x
end

# Fused RMSNorm + RoPE
function rmsnorm_rotary!(x::Matrix{Float32}, pos::Int, rope::RotaryEmbeddingCPU, norm::RMSNormCPU)
    head_dim, num_heads = size(x, 1), size(x, 2)
    half = div(rope.rotary_dim, 2)
    weight = norm.weight
    
    pos_idx = pos + 1
    @inbounds begin
        cos_vals = view(rope.cos_cache, :, pos_idx)
        sin_vals = view(rope.sin_cache, :, pos_idx)
    end
    
    for h in 1:num_heads
        # RMSNorm
        sum_sq = sum(abs2, view(x, :, h))
        rms = sqrt(sum_sq / head_dim + norm.eps)
        
        # Apply RMSNorm and RoPE
        @inbounds for i in 1:head_dim
            x[i, h] = x[i, h] / rms * weight[i]
        end
        
        # RoPE on all dimensions
        for i in 1:half
            idx1, idx2 = i, i + half
            x1, x2 = x[idx1, h], x[idx2, h]
            c, s = cos_vals[i], sin_vals[i]
            x[idx1, h] = x1 * c - x2 * s
            x[idx2, h] = x1 * s + x2 * c
        end
    end
    return x
end

# --- KV Cache ---
struct KVCacheCPU
    k::Array{Float32,3}
    v::Array{Float32,3}
end

function init_kv_cache_cpu(config::Qwen3ConfigCPU, max_seq::Int = 4096)
    k = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    v = zeros(Float32, config.head_dim, config.num_key_value_heads, max_seq)
    return KVCacheCPU(k, v)
end

function update_kv_cache!(cache::KVCacheCPU, k::Matrix{Float32}, v::Matrix{Float32}, pos::Int)
    cache.k[:, :, pos + 1] = k
    cache.v[:, :, pos + 1] = v
    return cache
end

# Weight type union
const QuantOrFloat32 = Union{Matrix{Float32}, Matrix{BFloat16}, Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

# --- MLP ---
struct MLPCPU
    gate_weight::QuantOrFloat32
    up_weight::QuantOrFloat32
    down_weight::QuantOrFloat32
    gate_buf::Vector{Float32}
    up_buf::Vector{Float32}
    hidden_buf::Vector{Float32}
    output_buf::Vector{Float32}
end

function mlp_forward(mlp::MLPCPU, x::Vector{Float32})
    # Gate + SiLU
    mul!(mlp.gate_buf, mlp.gate_weight, x)
    @turbo for i in eachindex(mlp.gate_buf)
        g = mlp.gate_buf[i]
        mlp.gate_buf[i] = g / (1.0f0 + exp(-g))
    end
    
    # Up
    mul!(mlp.up_buf, mlp.up_weight, x)
    
    # Element-wise multiply (SwiGLU)
    @turbo for i in eachindex(mlp.hidden_buf)
        mlp.hidden_buf[i] = mlp.gate_buf[i] * mlp.up_buf[i]
    end
    
    # Down
    mul!(mlp.output_buf, mlp.down_weight, mlp.hidden_buf)
    return mlp.output_buf
end

# Generic mat-vec multiplication
function mul!(out::Vector{Float32}, weight::QuantOrFloat32, x::Vector{Float32})
    if weight isa Matrix{Float32}
        LinearAlgebra.mul!(out, weight, x)
    elseif weight isa Matrix{BFloat16}
        ArrowLake.bf16_matmul_vec!(out, weight, x)
    else
        mul_quant_mat_vec(weight, x, out)
    end
    return out
end

# --- L2 Normalization ---
function l2norm!(x::AbstractArray{Float32}; eps::Float32 = 1e-6f0)
    norm_val = sqrt(sum(abs2, x) + eps)
    @. x = x / norm_val
    return x
end

# --- Full Attention (Pure attention, no SSM) ---
struct FullAttentionCPU
    q_weight::QuantOrFloat32
    k_weight::QuantOrFloat32
    v_weight::QuantOrFloat32
    o_weight::QuantOrFloat32
    q_norm::RMSNormCPU
    k_norm::RMSNormCPU
    q_buf::Matrix{Float32}
    k_buf::Matrix{Float32}
    v_buf::Matrix{Float32}
    cached_kv::Array{Float32,3}  # For efficient attention
    output_buf::Vector{Float32}
end

function (attn::FullAttentionCPU)(
    x::Vector{Float32},
    pos::Int,
    cache::KVCacheCPU,
    rope::RotaryEmbeddingCPU
)
    n_heads = size(attn.q_buf, 2)
    n_kv_heads = size(attn.k_buf, 2)
    head_dim = size(attn.q_buf, 1)
    scale = 1.0f0 / sqrt(head_dim)
    
    # Project Q, K, V
    mul!(view(attn.q_buf, :, :), attn.q_weight, x)
    mul!(view(attn.k_buf, :, :), attn.k_weight, x)
    mul!(view(attn.v_buf, :, :), attn.v_weight, x)
    
    # L2 normalize Q/K (critical for Qwen3)
    for h in 1:n_heads
        l2norm!(view(attn.q_buf, :, h))
    end
    for h in 1:n_kv_heads
        l2norm!(view(attn.k_buf, :, h))
    end
    
    # Apply RoPE to Q/K
    apply_rotary_emb!(attn.q_buf, pos, rope)
    apply_rotary_emb!(attn.k_buf, pos, rope)
    
    # Update KV cache
    update_kv_cache!(cache, attn.k_buf, attn.v_buf, pos)
    
    # Attention computation
    fill!(attn.output_buf, 0.0f0)
    seq_len = pos + 1
    
    for h in 1:n_heads
        # GQA: map query head to KV head
        kv_head = div((h - 1) * n_kv_heads, n_heads) + 1
        q = view(attn.q_buf, :, h)
        
        if h == 1
            # First head: accumulate output
            for t in 1:seq_len
                k = view(cache.k, :, kv_head, t)
                v = view(cache.v, :, kv_head, t)
                score = dot(q, k) * scale
                attn_score = exp(score)
                @. attn.output_buf += v * attn_score
            end
        else
            # Other heads
            for t in 1:seq_len
                k = view(cache.k, :, kv_head, t)
                v = view(cache.v, :, kv_head, t)
                score = dot(q, k) * scale
                attn_score = exp(score)
                @. attn.output_buf += v * attn_score
            end
        end
    end
    
    # Normalize attention output (simplified softmax)
    sum_out = sum(attn.output_buf)
    if sum_out > 0
        attn.output_buf ./= sum_out
    end
    
    # Final projection
    o_out = zero.(attn.output_buf)  # Reuse buffer
    mul!(o_out, attn.o_weight, attn.output_buf)
    
    return o_out
end

# --- Decoder Layer ---
struct DecoderLayerCPU
    attn::FullAttentionCPU
    mlp::MLPCPU
    attn_norm::RMSNormCPU
    mlp_norm::RMSNormCPU
    # Pre-allocated buffers
    residual_buf::Vector{Float32}
    norm_buf::Vector{Float32}
end

function (layer::DecoderLayerCPU)(
    x::Vector{Float32},
    pos::Int,
    rope::RotaryEmbeddingCPU,
    cache::KVCacheCPU
)
    # Pre-norm attention
    layer.residual_buf .= x
    rmsnorm_cpu!(layer.norm_buf, x, layer.attn_norm)
    
    # Attention
    attn_out = layer.attn(layer.norm_buf, pos, cache, rope)
    
    # Residual
    x .= layer.residual_buf
    x += attn_out
    
    # Pre-norm MLP
    rmsnorm_cpu!(layer.norm_buf, x, layer.mlp_norm)
    mlp_out = mlp_forward(layer.mlp, layer.norm_buf)
    
    # Final residual
    x += mlp_out
    
    return x
end

# --- Full Model ---
struct Qwen3ModelCPU
    config::Qwen3ConfigCPU
    embed_tokens::Matrix{Float32}
    lm_head::QuantOrFloat32
    layers::Vector{DecoderLayerCPU}
    norm::RMSNormCPU
    rope::RotaryEmbeddingCPU
    # Buffers
    hidden_buf::Vector{Float32}
    norm_buf::Vector{Float32}
    logits_buf::Vector{Float32}
end

# Forward pass
function forward_cpu!(
    model::Qwen3ModelCPU,
    tokens::Vector{Int},
    pos::Int,
    caches::Vector{KVCacheCPU}
)
    seq_len = length(tokens)
    x = model.hidden_buf
    
    # Embed first token
    for i in eachindex(x)
        x[i] = model.embed_tokens[tokens[1] + 1, i]  # Assuming 0-indexed vocab
    end
    
    # Pass through all layers
    for (i, layer) in enumerate(model.layers)
        x = layer(x, pos, model.rope, caches[i])
    end
    
    # Final norm
    rmsnorm_cpu!(model.norm_buf, x, model.norm)
    
    # LM head
    lm_head_project!(model.logits_buf, model.lm_head, model.norm_buf)
    
    return model.logits_buf
end

function lm_head_project!(output::Vector{Float32}, weight::QuantOrFloat32, hidden::Vector{Float32})
    if weight isa Matrix{Float32}
        LinearAlgebra.mul!(output, weight, hidden)
    elseif weight isa Matrix{BFloat16}
        ArrowLake.bf16_matmul_vec!(output, weight, hidden)
    else
        mul_quant_mat_vec(weight, hidden, output)
    end
    return output
end

# --- Generation ---
function generate_cpu(
    model::Qwen3ModelCPU,
    tokenizer,
    prompt::String;
    max_tokens::Int = 50,
    temperature::Float32 = 1.0f0
)
    input_ids = tokenizer.encode(prompt)
    
    # Initialize KV caches
    caches = [init_kv_cache_cpu(model.config) for _ in model.layers]
    
    # Generate
    output_tokens = Int[]
    
    for i in 1:min(max_tokens, model.config.max_position_embeddings - length(input_ids) - 1)
        pos = length(input_ids) + i - 1
        
        # Forward pass
        logits = forward_cpu!(model, vcat(input_ids, output_tokens), pos, caches)
        
        # Sample next token
        probs = CommonOps.softmax(logits)
        next_token = CommonOps.softmax_sample(probs; temperature=temperature)
        
        push!(output_tokens, next_token)
        
        # Check for EOS
        if next_token == model.config.eos_token_id
            break
        end
    end
    
    return tokenizer.decode(output_tokens)
end

# --- Model Display ---
function Base.show(io::IO, ::MIME"text/plain", model::Qwen3ModelCPU)
    config = model.config
    println(io, "Qwen3ModelCPU (CPU backend)")
    println(io, "├─ Architecture: ", config.architecture)
    println(io, "├─ Hidden size: ", config.hidden_size)
    println(io, "├─ Layers: ", config.num_hidden_layers)
    println(io, "├─ Attention heads: ", config.num_attention_heads, " (KV: ", config.num_key_value_heads, ")")
    println(io, "├─ Head dim: ", config.head_dim)
    println(io, "├─ Vocab size: ", config.vocab_size)
    println(io, "└─ Max positions: ", config.max_position_embeddings)
end

# --- Qwen3 GGUF Loader ---
# --- Qwen3 GGUF Loader ---
function load_qwen3_cpu(path::String; keep_quantized::Union{Bool,Nothing}=nothing, use_bf16_weights::Bool=false)
    @info "Loading Qwen3 model from: $path"
    file = GGUF.read_gguf(path)
    @info "Qwen3 GGUF loaded" tensors=length(file.tensors)
    @info "Qwen3 GGUF loaded" tensors=length(file.tensors)
    
    # Create config from metadata
    config = Qwen3ConfigCPU(
        architecture=:qwen3,
        vocab_size=get(file.metadata, "qwen3.vocab_size", 151936),
        hidden_size=get(file.metadata, "qwen3.embedding_length", 2560),
        intermediate_size=get(file.metadata, "qwen3.feed_forward_length", 9728),
        num_hidden_layers=get(file.metadata, "qwen3.block_count", 36),
        num_attention_heads=get(file.metadata, "qwen3.attention.head_count", 32),
        num_key_value_heads=get(file.metadata, "qwen3.attention.head_count_kv", 8),
        head_dim=get(file.metadata, "qwen3.attention.key_length", 128),
        rms_norm_eps=Float32(get(file.metadata, "qwen3.attention.layer_norm_rms_epsilon", 1e-6)),
        rope_theta=Float32(get(file.metadata, "qwen3.rope.freq_base", 1000000.0)),
        max_position_embeddings=get(file.metadata, "qwen3.context_length", 32768)
    )
    
    @info "Qwen3 config" hidden=config.hidden_size layers=config.num_hidden_layers heads=config.num_attention_heads
    
    # Load embedding
    embed = Float32.(GGUF.extract_tensor(file, "token_embd.weight"))
    @info "Embedding loaded" shape=size(embed)
    
    # Load layers  
    layers = DecoderLayerCPU[]
    for i in 0:(config.num_hidden_layers - 1)
        layer = load_qwen3_layer(file, i, config; keep_quantized=keep_quantized)
        push!(layers, layer)
    end
    
    # Load final norm
    final_norm_w = Float32.(GGUF.extract_tensor(file, "output_norm.weight"))
    final_norm = RMSNormCPU(final_norm_w, config.rms_norm_eps)
    
    # LM head (tied with embedding for Qwen3)
    lm_head = embed'
    
    # Create RoPE with full rotary (all head_dim)
    rope = RotaryEmbeddingCPU(config.head_dim, config.rope_theta, config.max_position_embeddings)
    
    # Pre-allocate buffers
    hidden_buf = Vector{Float32}(undef, config.hidden_size)
    norm_buf = Vector{Float32}(undef, config.hidden_size)
    logits_buf = Vector{Float32}(undef, config.vocab_size)
    
    model = Qwen3ModelCPU(config, embed, lm_head, layers, final_norm, rope, hidden_buf, norm_buf, logits_buf)
    tok = Tokenizer.load_tokenizer(file.metadata)
    
    @info "Qwen3 model loaded successfully"
    return model, tok
end

function load_qwen3_layer(file, layer_idx::Int, config::Qwen3ConfigCPU; keep_quantized::Union{Bool,Nothing}=nothing)
    prefix = "blk.$(layer_idx)"
    
    # Load norms
    attn_norm_w = Float32.(GGUF.extract_tensor(file, "$(prefix).attn_norm.weight"))
    attn_norm = RMSNormCPU(attn_norm_w, config.rms_norm_eps)
    
    mlp_norm_w = Float32.(GGUF.extract_tensor(file, "$(prefix).mlp_norm.weight"))
    mlp_norm = RMSNormCPU(mlp_norm_w, config.rms_norm_eps)
    
    # Load attention weights
    q_proj_w = GGUF.extract_tensor(file, "$(prefix).attn_q.weight")
    k_proj_w = GGUF.extract_tensor(file, "$(prefix).attn_k.weight")
    v_proj_w = GGUF.extract_tensor(file, "$(prefix).attn_v.weight")
    o_proj_w = GGUF.extract_tensor(file, "$(prefix).attn_o.weight")
    
    # Load MLP weights
    gate_proj_w = GGUF.extract_tensor(file, "$(prefix).mlp_gate.weight")
    up_proj_w = GGUF.extract_tensor(file, "$(prefix).mlp_up.weight")
    down_proj_w = GGUF.extract_tensor(file, "$(prefix).mlp_down.weight")
    
    # Create attention buffers  
    q_buf = Matrix{Float32}(undef, config.num_attention_heads, config.head_dim)
    k_buf = Matrix{Float32}(undef, config.num_key_value_heads, config.head_dim)
    v_buf = Matrix{Float32}(undef, config.num_key_value_heads, config.head_dim)
    output_buf = Vector{Float32}(undef, config.hidden_size)
    
    attn = FullAttentionCPU(
        q_proj_w, k_proj_w, v_proj_w, o_proj_w,
        RMSNormCPU(zeros(config.head_dim), 1e-6), RMSNormCPU(zeros(config.head_dim), 1e-6),
        q_buf, k_buf, v_buf, zeros(Float32, config.head_dim, config.num_attention_heads, 4096),
        output_buf
    )
    
    # Create MLP
    mlp = MLPCPU(gate_proj_w, up_proj_w, down_proj_w,
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.hidden_size))
    
    # Layer buffers
    residual_buf = Vector{Float32}(undef, config.hidden_size)
    norm_buf = Vector{Float32}(undef, config.hidden_size)
    
    return DecoderLayerCPU(attn, mlp, attn_norm, mlp_norm, residual_buf, norm_buf)
end

# --- Qwen3 Safetensors Loader ---
function load_qwen3_safetensors(path::String; keep_quantized::Union{Bool,Nothing}=nothing, use_bf16_weights::Bool=false)
    @info "Loading Qwen3 safetensors model from: $path"
    file = Safetensors.parse_safetensors(path)
    @info "Qwen3 safetensors loaded" tensors=length(file)
    
    sf = parse_safetensors(model_path)
    @info "Qwen3 safetensors loaded" tensors=length(sf.tensors)
    
    # Load config.json
    config_path = joinpath(path, "config.json")
    if !isfile(config_path)
        error("config.json not found: $config_path")
    end
    config_json = JSON.parsefile(config_path)
    text_config = get(config_json, "text_config", config_json)
    
    # Create config
    config = Qwen3ConfigCPU(
        architecture=:qwen3,
        vocab_size=get(text_config, "vocab_size", 151936),
        hidden_size=get(text_config, "hidden_size", 2560),
        intermediate_size=get(text_config, "intermediate_size", 9728),
        num_hidden_layers=get(text_config, "num_hidden_layers", 36),
        num_attention_heads=get(text_config, "num_attention_heads", 32),
        num_key_value_heads=get(text_config, "num_key_value_heads", 8),
        head_dim=get(text_config, "head_dim", 128),
        rms_norm_eps=Float32(get(text_config, "rms_norm_eps", 1e-6)),
        rope_theta=Float32(get(text_config, "rope_theta", 1000000.0)),
        max_position_embeddings=get(text_config, "max_position_embeddings", 32768)
    )
    
    @info "Qwen3 config" hidden=config.hidden_size layers=config.num_hidden_layers heads=config.num_attention_heads
    
    # Load embedding - Qwen3 uses "model.embed_tokens.weight"
    embed_raw = get_tensor(sf, "model.embed_tokens.weight")
    if embed_raw === nothing
        error("embed_tokens.weight not found")
    end
    embed = Matrix(Float32.(embed_raw'))  # Transpose to (hidden, vocab)
    @info "Embedding loaded" shape=size(embed)
    
    # Load layers
    layers = DecoderLayerCPU[]
    for i in 0:(config.num_hidden_layers - 1)
        println("Loading Qwen3 layer $i...")
        layer = load_qwen3_layer_safetensors(sf, i, config)
        push!(layers, layer)
    end
    
    # Load final norm
    final_norm_raw = get_tensor(sf, "model.norm.weight")
    if final_norm_raw === nothing
        error("norm.weight not found")
    end
    final_norm = RMSNormCPU(vec(Float32.(final_norm_raw)), config.rms_norm_eps)
    
    # LM head (tied with embedding if not present)
    lm_head_key = "lm_head.weight"
    if haskey(sf.tensors, lm_head_key)
        lm_head_raw = get_tensor(sf, lm_head_key)
        lm_head = Matrix(Float32.(lm_head_raw'))
    else
        # Tied weights
        lm_head = embed'
    end
    
    # Create RoPE with full rotary
    rope = RotaryEmbeddingCPU(config.head_dim, config.rope_theta, config.max_position_embeddings)
    
    # Pre-allocate buffers
    hidden_buf = Vector{Float32}(undef, config.hidden_size)
    norm_buf = Vector{Float32}(undef, config.hidden_size)
    logits_buf = Vector{Float32}(undef, config.vocab_size)
    
    model = Qwen3ModelCPU(config, embed, lm_head, layers, final_norm, rope, hidden_buf, norm_buf, logits_buf)
    
    # Load tokenizer
    tok = Tokenizer.load_tokenizer(path)
    
    @info "Qwen3 safetensors model loaded successfully"
    return model, tok
end

function load_qwen3_layer_safetensors(sf, layer_idx::Int, config::Qwen3ConfigCPU)
    prefix = "model.layers.$(layer_idx)"
    
    # Load norms
    attn_norm_w = get_tensor(sf, "$(prefix).input_layernorm.weight")
    if attn_norm_w === nothing
        error("input_layernorm.weight not found for layer $layer_idx")
    end
    attn_norm = RMSNormCPU(vec(Float32.(attn_norm_w)), config.rms_norm_eps)
    
    mlp_norm_w = get_tensor(sf, "$(prefix).post_attention_layernorm.weight")
    mlp_norm = RMSNormCPU(vec(Float32.(mlp_norm_w)), config.rms_norm_eps)
    
    # Load attention weights
    q_proj = get_tensor(sf, "$(prefix).self_attn.q_proj.weight")
    k_proj = get_tensor(sf, "$(prefix).self_attn.k_proj.weight")
    v_proj = get_tensor(sf, "$(prefix).self_attn.v_proj.weight")
    o_proj = get_tensor(sf, "$(prefix).self_attn.o_proj.weight")
    
    # Load MLP weights
    gate_proj = get_tensor(sf, "$(prefix).mlp.gate_proj.weight")
    up_proj = get_tensor(sf, "$(prefix).mlp.up_proj.weight")
    down_proj = get_tensor(sf, "$(prefix).mlp.down_proj.weight")
    
    # Create attention buffers
    q_buf = Matrix{Float32}(undef, config.num_attention_heads, config.head_dim)
    k_buf = Matrix{Float32}(undef, config.num_key_value_heads, config.head_dim)
    v_buf = Matrix{Float32}(undef, config.num_key_value_heads, config.head_dim)
    output_buf = Vector{Float32}(undef, config.hidden_size)
    
    attn = FullAttentionCPU(
        Float32.(q_proj'), Float32.(k_proj'), Float32.(v_proj'), Float32.(o_proj'),
        RMSNormCPU(zeros(config.head_dim), 1e-6), RMSNormCPU(zeros(config.head_dim), 1e-6),
        q_buf, k_buf, v_buf, zeros(Float32, config.head_dim, config.num_attention_heads, 4096),
        output_buf
    )
    
    # Create MLP
    mlp = MLPCPU(Float32.(gate_proj'), Float32.(up_proj'), Float32.(down_proj'),
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.intermediate_size),
                 zeros(Float32, config.hidden_size))
    
    # Layer buffers
    residual_buf = Vector{Float32}(undef, config.hidden_size)
    norm_buf = Vector{Float32}(undef, config.hidden_size)
    
    return DecoderLayerCPU(attn, mlp, attn_norm, mlp_norm, residual_buf, norm_buf)
end

end  # module Qwen3