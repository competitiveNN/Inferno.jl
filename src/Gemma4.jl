module Gemma4

using LinearAlgebra
using Printf
using ..CommonOps

# ============================================================
# Gemma4 CPU Inference Implementation
# ============================================================
#
# Key architectural features:
# 1. Mixed sliding/full attention layers
# 2. Proportional + partial RoPE for full attention (25%, theta=1M)
# 3. Default RoPE for sliding attention (full rotary, theta=10K)
# 4. Per-layer input embeddings (unique to Gemma4)
# 5. KV cache sharing across layers
# 6. Attention logit softcapping
# 7. Final logit softcapping (tanh cap at 30.0)
# 8. Q/K RMSNorm (with_scale=True), V RMSNorm (with_scale=False)
# 9. GELU tanh activation (not SiLU)
# 10. Embedding scaling (multiply by sqrt(hidden_size))
# 11. Layer scalar (learnable per-layer scale, applied to whole hidden)
# 12. attention_k_eq_v=false (V has its own projection)
# 13. scaling=1.0 for attention (Q/K norm handles scaling)
# ============================================================
#
# Weight convention (CRITICAL):
# Safetensors stores weights as (out_features, in_features).
# get_tensor() preserves this: W is (out, in).
# Therefore: y = W * x (NOT W' * x)
# ============================================================

# --- Structs ---

struct Gemma4Config
    hidden_size::Int
    num_layers::Int
    num_q_heads::Int
    num_kv_heads::Int
    num_global_kv_heads::Int
    head_dim::Int
    global_head_dim::Int
    intermediate_size::Int
    double_wide_intermediate::Int
    vocab_size::Int
    vocab_size_per_layer_input::Int
    max_seq_len::Int
    sliding_window::Int
    rms_norm_eps::Float32
    final_logit_softcapping::Float32
    attention_logits_softcapping::Float32
    embed_scale::Float32
    per_layer_input_scale::Float32
    per_layer_model_projection_scale::Float32
    layer_types::Vector{String}
    num_kv_shared_layers::Int
    first_kv_shared_layer::Int # index of first shared layer (0-based)
    hidden_size_per_layer_input::Int
    attention_k_eq_v::Bool
    # RoPE params
    sliding_rope_theta::Float32
    full_rope_theta::Float32
    full_partial_rotary_factor::Float32
    tie_word_embeddings::Bool
    # Which layers should store full KV for sharing (0-based indices)
    store_kv_layers::Vector{Int}
end

mutable struct AttentionLayer
    q_proj::Matrix{Float32}       # (num_q_heads * head_dim, hidden_size) — W*x gives (out,)
    k_proj::Matrix{Float32}       # (num_kv_heads * head_dim, hidden_size) — empty for shared KV layers
    v_proj::Matrix{Float32}       # (num_kv_heads * head_dim, hidden_size) — empty if k_eq_v or shared
    o_proj::Matrix{Float32}       # (hidden_size, num_q_heads * head_dim)
    q_norm_w::Vector{Float32}     # (head_dim,) RMSNorm with scale
    k_norm_w::Vector{Float32}     # (head_dim,) RMSNorm with scale — empty for shared KV layers
    is_sliding::Bool
    is_kv_shared::Bool
    kv_shared_src::Int            # source layer index for shared KV (0-based)
    head_dim::Int                 # actual head_dim for this layer (sliding vs global)
    num_kv_heads_actual::Int      # num_kv_heads for this layer
    # Pre-allocated buffers
    q_buf::Vector{Float32}
    k_buf::Vector{Float32}
    v_buf::Vector{Float32}
    attn_out_buf::Vector{Float32}
end

mutable struct MLPLayer
    gate_proj::Matrix{Float32}    # (intermediate, hidden) — W*x
    up_proj::Matrix{Float32}      # (intermediate, hidden) — W*x
    down_proj::Matrix{Float32}    # (hidden, intermediate) — W*x
    # Pre-allocated buffers
    gate_buf::Vector{Float32}
    up_buf::Vector{Float32}
    hidden_buf::Vector{Float32}
end

mutable struct PerLayerInput
    gate_proj::Matrix{Float32}    # (pli_size, hidden) — W*x
    projection::Matrix{Float32}   # (hidden, pli_size) — W*x
    post_norm_w::Vector{Float32}  # (hidden,) RMSNorm with scale
end

mutable struct DecoderLayer
    input_norm_w::Vector{Float32}     # input_layernorm
    post_attn_norm_w::Vector{Float32} # post_attention_layernorm
    pre_ff_norm_w::Vector{Float32}    # pre_feedforward_layernorm
    post_ff_norm_w::Vector{Float32}   # post_feedforward_layernorm
    attn::AttentionLayer
    mlp::MLPLayer
    pli::Union{PerLayerInput, Nothing}
    layer_scalar::Float32
    # Pre-allocated buffers
    norm_buf::Vector{Float32}
    pli_gate_buf::Vector{Float32}
    pli_out_buf::Vector{Float32}
end

struct KVCacheG4
    k_cache::Vector{Matrix{Float32}} # one per layer: (num_kv_heads * head_dim, max_seq_len)
    v_cache::Vector{Matrix{Float32}} # one per layer
    seq_len::Vector{Int}            # current length per layer
end

mutable struct Gemma4Model
    config::Gemma4Config
    embed_tokens::Matrix{Float32}              # (vocab_size, hidden_size)
    embed_tokens_per_layer::Matrix{Float32}    # (vocab_per_layer, num_layers * pli_size)
    per_layer_model_proj::Matrix{Float32}      # (num_layers * pli_size, hidden_size) — W*x gives (num_layers*pli_size,)
    per_layer_proj_norm_w::Vector{Float32}     # (pli_size,) RMSNorm
    final_norm_w::Vector{Float32}              # (hidden_size,) final RMSNorm
    layers::Vector{DecoderLayer}
    # RoPE pre-computed
    sliding_cos::Vector{Vector{Float32}} # per position: (head_dim/2,)
    sliding_sin::Vector{Vector{Float32}}
    full_cos::Vector{Vector{Float32}}    # per position: (global_head_dim/2,)
    full_sin::Vector{Vector{Float32}}
    # Shared KV states (populated during prefill for store_kv_layers)
    shared_kv_k::Dict{Int, Matrix{Float32}} # 0-based layer_idx => (num_kv_heads * head_dim, max_seq_len)
    shared_kv_v::Dict{Int, Matrix{Float32}}
    # Pre-allocated buffers
    hidden_buf::Vector{Float32}               # (hidden_size,)
    residual_buf::Vector{Float32}             # (hidden_size,)
    pli_embed_buf::Vector{Float32}            # (num_layers * pli_size,) — combined per-layer inputs
    pli_proj_buf::Vector{Float32}             # (num_layers * pli_size,) — projection output
    logits_buf::Vector{Float32}               # (vocab_size,)
    # Temp for logits computation
    logits_hidden_scaled::Vector{Float32}     # (hidden_size,) — hidden * embed_scale for tied lm_head
end

# --- KV Cache ---

function init_kv_cache(config::Gemma4Config, max_seq_len::Int)
    k_caches = Matrix{Float32}[]
    v_caches = Matrix{Float32}[]
    seq_lens = Int[]
    for i in 1:config.num_layers
        head_d = config.head_dim
        n_kv = config.num_kv_heads
        # For full attention layers that use global head dim
        if config.layer_types[i] == "full_attention"
            head_d = config.global_head_dim
            n_kv = config.num_global_kv_heads
        end
        push!(k_caches, Matrix{Float32}(undef, n_kv * head_d, max_seq_len))
        push!(v_caches, Matrix{Float32}(undef, n_kv * head_d, max_seq_len))
        push!(seq_lens, 0)
    end
    return KVCacheG4(k_caches, v_caches, seq_lens)
end

# --- Determine which layers store full KV for sharing ---

function compute_store_kv_layers(config::Gemma4Config)
    """Find the last non-shared layer of each type — these store full KV for sharing."""
    store_layers = Int[]
    if config.num_kv_shared_layers == 0
        return store_layers
    end
    first_shared = config.first_kv_shared_layer # 0-based
    # prev_layers = layers before first_shared (0-based)
    prev_layer_types = config.layer_types[1:first_shared] # Julia 1-based slice

    # Find unique types among shared layers
    shared_types = Set{String}()
    for i in (first_shared+1):config.num_layers
        push!(shared_types, config.layer_types[i])
    end

    for lt in shared_types
        # Find last non-shared layer of this type (search backward)
        for j in first_shared:-1:1
            if config.layer_types[j] == lt
                push!(store_layers, j - 1) # store as 0-based
                break
            end
        end
    end
    return sort(store_layers)
end

# --- RoPE ---

function precompute_rope(config::Gemma4Config, max_seq_len::Int)
    # Sliding attention: default RoPE, full rotary, theta=10000
    sliding_dim = config.head_dim
    sliding_half = sliding_dim ÷ 2
    sliding_theta = Float64(config.sliding_rope_theta)
    sliding_inv_freq = [1.0 / (sliding_theta ^ (2.0 * k / sliding_dim)) for k in 0:(sliding_half - 1)]

    # Full attention: proportional RoPE, partial=25%, theta=1M
    full_dim = config.global_head_dim
    full_half = full_dim ÷ 2
    full_theta = Float64(config.full_rope_theta)
    partial_factor = config.full_partial_rotary_factor
    rope_angles = Int(partial_factor * full_dim ÷ 2) # number of rotating pairs

    # Proportional RoPE: exponent uses full head_dim, not partial
    full_inv_freq = Vector{Float64}(undef, full_half)
    for k in 0:(rope_angles - 1)
        # k ranges over pairs, exponent = 2*k / full_dim
        full_inv_freq[k + 1] = 1.0 / (full_theta ^ (2.0 * k / full_dim))
    end
    # Non-rotated portion gets zero frequency (cos=1, sin=0)
    for k in rope_angles:full_half-1
        full_inv_freq[k + 1] = 0.0
    end

    # Pre-compute cos/sin for all positions
    sliding_cos = Vector{Float32}[]
    sliding_sin = Vector{Float32}[]
    full_cos = Vector{Float32}[]
    full_sin = Vector{Float32}[]

    for pos in 0:(max_seq_len - 1)
        sc = Float32[cos(pos * sliding_inv_freq[k+1]) for k in 0:sliding_half-1]
        ss = Float32[sin(pos * sliding_inv_freq[k+1]) for k in 0:sliding_half-1]
        push!(sliding_cos, sc)
        push!(sliding_sin, ss)

        fc = Float32[cos(pos * full_inv_freq[k+1]) for k in 0:full_half-1]
        fs = Float32[sin(pos * full_inv_freq[k+1]) for k in 0:full_half-1]
        push!(full_cos, fc)
        push!(full_sin, fs)
    end

    return sliding_cos, sliding_sin, full_cos, full_sin
end

# --- Math functions ---

function gelu_tanh(x::Float32)
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = sqrt(2.0f0 / Float32(pi))
    return 0.5f0 * x * (1.0f0 + tanh(c * (x + 0.044715f0 * x * x * x)))
end

# Re-export CommonOps functions used throughout Gemma4
const rmsnorm! = CommonOps.rmsnorm!
const rmsnorm_no_scale! = CommonOps.rmsnorm_no_scale!

function apply_rope!(q::AbstractVector{Float32}, pos::Int,
    cos_vals::AbstractVector{Float32},
    sin_vals::AbstractVector{Float32},
    head_dim::Int, num_heads::Int; rotary_dim::Int=0)
# HuggingFace rotate_half RoPE convention:
# Split head dims into first-half and second-half at head_dim/2.
# Pair x[k] with x[k + head_dim/2], NOT consecutive (x[2k], x[2k+1]).
# out[k] = x[k] * cos[k] - x[k+half] * sin[k]
# out[k + half] = x[k] * sin[k] + x[k+half] * cos[k]
# For partial RoPE, cos/sin have length head_dim/2, with cos=1,sin=0 for
# non-rotary dims (indices rotary_dim/2+1 to head_dim/2).
# rotary_dim controls which dims have non-trivial rotation (default: head_dim).
if rotary_dim <= 0
    rotary_dim = head_dim
end
half = head_dim ÷ 2  # ALWAYS split at head_dim/2, not rotary_dim/2!
for h in 0:(num_heads - 1)
    base = h * head_dim + 1
    for k in 1:half
        c = cos_vals[k]
        s = sin_vals[k]
        idx1 = base + k - 1 # first half element
        idx2 = base + k - 1 + half # second half element
        q0 = q[idx1]
        q1 = q[idx2]
        q[idx1] = q0 * c - q1 * s
        q[idx2] = q0 * s + q1 * c
    end
end
end

# --- Attention forward ---

function attention_forward!(attn::AttentionLayer, hidden::AbstractVector{Float32},
                            pos::Int, cache::KVCacheG4, layer_idx::Int,
                            sliding_cos::Vector{Vector{Float32}},
                            sliding_sin::Vector{Vector{Float32}},
                            full_cos::Vector{Vector{Float32}},
                            full_sin::Vector{Vector{Float32}},
                            shared_kv_k::Dict{Int, Matrix{Float32}},
                            shared_kv_v::Dict{Int, Matrix{Float32}},
                            config::Gemma4Config)
    head_d = attn.head_dim
    n_q = config.num_q_heads
    n_kv = attn.num_kv_heads_actual
    softcap = config.attention_logits_softcapping
    is_sliding = attn.is_sliding

    # Q projection + norm + RoPE
    # q_proj is (out, in), so W * x gives (out,)
    mul!(attn.q_buf, attn.q_proj, hidden)
    # Q norm (RMSNorm with scale) — per head
    for h in 0:(n_q - 1)
        base = h * head_d + 1
        q_head = view(attn.q_buf, base:base+head_d-1)
        rmsnorm!(q_head, q_head, attn.q_norm_w, config.rms_norm_eps)
    end
 # Apply RoPE to Q
 if is_sliding
  apply_rope!(attn.q_buf, pos, sliding_cos[pos+1], sliding_sin[pos+1], head_d, n_q)
 else
  # Full attention: partial rotary (only first 25% of dims get RoPE)
  full_rotary_dim = config.global_head_dim ÷ 4  # 512 * 0.25 = 128
  apply_rope!(attn.q_buf, pos, full_cos[pos+1], full_sin[pos+1], head_d, n_q; rotary_dim=full_rotary_dim)
 end

    # K/V handling
 if attn.is_kv_shared
 # Use shared KV from source layer (0-based index)
 # The source layer stores FULL K/V in shared_kv_k/v.
 # We need to copy ALL positions up to current into this layer's cache.
 src = attn.kv_shared_src
 k_states = shared_kv_k[src]
 v_states = shared_kv_v[src]
 num_positions = size(k_states, 2)  # all positions stored by source
 cache.k_cache[layer_idx][:, 1:num_positions] = k_states
 cache.v_cache[layer_idx][:, 1:num_positions] = v_states
    else
 # K projection
 mul!(attn.k_buf, attn.k_proj, hidden) # W*x gives (n_kv * head_d,)

 # V projection + norm (no scale) — MUST be done BEFORE K norm/RoPE
 # In HF: value_states = v_proj(hidden) if v_proj else key_states (raw K before norm/rope)
 # Then v_norm(value_states) with with_scale=False
 if size(attn.v_proj, 1) > 0
 mul!(attn.v_buf, attn.v_proj, hidden) # W*x
 else
 # k_eq_v mode: V = raw K (before norm/rope was applied)
 # In HF: value_states = key_states where key_states = k_proj(hidden).view(...) — RAW K
 copyto!(attn.v_buf, attn.k_buf)
 end
 # V norm (no scale)
 for h in 0:(n_kv - 1)
 base = h * head_d + 1
 v_head = view(attn.v_buf, base:base+head_d-1)
 rmsnorm_no_scale!(v_head, v_head, config.rms_norm_eps)
 end

 # K norm + RoPE — done AFTER V is saved from raw K
 for h in 0:(n_kv - 1)
 base = h * head_d + 1
 k_head = view(attn.k_buf, base:base+head_d-1)
 rmsnorm!(k_head, k_head, attn.k_norm_w, config.rms_norm_eps)
 end
 if is_sliding
  apply_rope!(attn.k_buf, pos, sliding_cos[pos+1], sliding_sin[pos+1], head_d, n_kv)
 else
  # Full attention: partial rotary (only first 25% of dims get RoPE)
  full_rotary_dim = config.global_head_dim ÷ 4  # 512 * 0.25 = 128
  apply_rope!(attn.k_buf, pos, full_cos[pos+1], full_sin[pos+1], head_d, n_kv; rotary_dim=full_rotary_dim)
 end

        # Store in cache
        cache.k_cache[layer_idx][:, pos+1] = attn.k_buf
        cache.v_cache[layer_idx][:, pos+1] = attn.v_buf
    end

    cache.seq_len[layer_idx] = max(cache.seq_len[layer_idx], pos + 1)
 # For KV shared layers, seq_len should reflect all shared positions
 if attn.is_kv_shared
 src = attn.kv_shared_src
 if haskey(shared_kv_k, src)
 cache.seq_len[layer_idx] = max(cache.seq_len[layer_idx], size(shared_kv_k[src], 2))
 end
 end

    # Compute attention: for each Q head, attend to all K/V positions
    seq_len = cache.seq_len[layer_idx]
    kv_group = n_q ÷ n_kv # GQA group size

    fill!(attn.attn_out_buf, 0.0f0)

    for h in 0:(n_q - 1)
        kv_h = h ÷ kv_group # which KV head this Q head attends to
        q_base = h * head_d + 1
        kv_base = kv_h * head_d + 1

        # Sliding window: determine valid attention range
        if is_sliding
            window_start = max(1, pos + 1 - config.sliding_window + 1)
        else
            window_start = 1
        end

        # Compute attention scores for valid range
        q_head = view(attn.q_buf, q_base:q_base+head_d-1)
        attn_start = window_start
        attn_end = pos + 1 # current position (causal)

        n_positions = attn_end - attn_start + 1
        scores = Vector{Float32}(undef, n_positions)
        for (ti, t) in enumerate(attn_start:attn_end)
            k_t = view(cache.k_cache[layer_idx], kv_base:kv_base+head_d-1, t)
            s = 0.0f0
            @simd for d in 1:head_d
                s += q_head[d] * k_t[d]
            end
            # Attention logit softcapping
            if softcap > 0
                s = tanh(s / softcap) * softcap
            end
            scores[ti] = s
        end

        # Softmax
        max_score = maximum(scores)
        sum_weights = 0.0f0
        @simd for i in 1:n_positions
            scores[i] = exp(scores[i] - max_score)
            sum_weights += scores[i]
        end
        inv_sum = 1.0f0 / sum_weights
        @simd for i in 1:n_positions
            scores[i] *= inv_sum
        end

        # Weighted sum of V
        out_base = h * head_d + 1
        for (ti, t) in enumerate(attn_start:attn_end)
            w = scores[ti]
            @simd for d in 1:head_d
                attn.attn_out_buf[out_base + d - 1] += w * cache.v_cache[layer_idx][kv_base + d - 1, t]
            end
        end
    end

    # Output projection: o_proj is (hidden_size, n_q * head_dim)
    # W * attn_out_buf gives (hidden_size,)
    result = hidden # reuse hidden as output buffer (caller saved residual already)
    mul!(result, attn.o_proj, attn.attn_out_buf)

    return result
end

# --- Per-layer input ---

function compute_per_layer_inputs!(model::Gemma4Model, token_id::Int, inputs_embeds::AbstractVector{Float32})
    config = model.config
    pli_size = config.hidden_size_per_layer_input
    n_layers = config.num_layers

    if pli_size == 0
        return
    end

 # Step 1: embed_tokens_per_layer lookup
 # embed_tokens_per_layer is (vocab_per_layer, num_layers * pli_size)
 # Get row for token_id (already 1-indexed from tokenizer)
 # CRITICAL: HF uses Gemma4TextScaledWordEmbedding which multiplies by embed_scale = sqrt(pli_size)
 # Our raw weights are NOT scaled, so we must apply the scaling here.
 pli_row = view(model.embed_tokens_per_layer, token_id, :)
 pli_scale = sqrt(Float32(pli_size))  # sqrt(256) = 16.0

 # Step 2: per_layer_model_projection
 # per_layer_model_proj is (num_layers * pli_size, hidden_size)
 # projection = proj * inputs_embeds * projection_scale
 mul!(model.pli_proj_buf, model.per_layer_model_proj, inputs_embeds)
 # Scale was already applied during loading (line 137 of Loader)

 # Step 3: Following HF project_per_layer_inputs():
 # per_layer_projection = per_layer_model_projection(inputs_embeds) * scale
 # per_layer_projection = reshape to (seq, num_layers, pli_size)
 # per_layer_projection = per_layer_projection_norm(per_layer_projection) ← NORM FIRST
 # result = (per_layer_projection + per_layer_inputs) * per_layer_input_scale ← THEN ADD + SCALE
 for i in 1:n_layers
 offset = (i - 1) * pli_size + 1
 proj_slice = view(model.pli_proj_buf, offset:offset+pli_size-1)
 embed_slice = view(pli_row, offset:offset+pli_size-1)
 out_slice = view(model.pli_embed_buf, offset:offset+pli_size-1)

 # First: norm the projection
 rmsnorm!(out_slice, proj_slice, model.per_layer_proj_norm_w, config.rms_norm_eps)

 # Then: (normed_proj + embed * pli_scale) * per_layer_input_scale
 @simd for d in 1:pli_size
 out_slice[d] = (out_slice[d] + embed_slice[d] * pli_scale) * config.per_layer_input_scale
 end
 end
end

function per_layer_input_forward!(layer::DecoderLayer, pli_slice::AbstractVector{Float32},
                                  hidden::AbstractVector{Float32}, config::Gemma4Config)
    pli = layer.pli
    if pli === nothing
        return
    end

    n = length(hidden)
    pli_size = config.hidden_size_per_layer_input

    # gate = gate_proj(hidden) → gelu → gate * per_layer_input → projection → norm
    # gate_proj is (pli_size, hidden), W*x gives (pli_size,)
    mul!(layer.pli_gate_buf, pli.gate_proj, hidden)

    # gate = gelu_tanh(gate)
    @simd for i in 1:pli_size
        layer.pli_gate_buf[i] = gelu_tanh(layer.pli_gate_buf[i])
    end

    # gate = gate * per_layer_input
    @simd for i in 1:pli_size
        layer.pli_gate_buf[i] *= pli_slice[i]
    end

    # out = projection(gate * pli)
    # projection is (hidden, pli_size), W*x gives (hidden,)
    mul!(layer.pli_out_buf, pli.projection, layer.pli_gate_buf)

    # post_per_layer_input_norm (RMSNorm with scale)
    rmsnorm!(layer.pli_out_buf, layer.pli_out_buf, pli.post_norm_w, config.rms_norm_eps)
end

# --- MLP forward ---

function mlp_forward!(layer::DecoderLayer, hidden::AbstractVector{Float32}, config::Gemma4Config)
    mlp = layer.mlp
    n = length(hidden)
    inter = size(mlp.gate_proj, 1)

    # gate = gelu_tanh(gate_proj(hidden))
    # gate_proj is (intermediate, hidden), W*x gives (intermediate,)
    mul!(mlp.gate_buf, mlp.gate_proj, hidden)
    @simd for i in 1:inter
        mlp.gate_buf[i] = gelu_tanh(mlp.gate_buf[i])
    end

    # up = up_proj(hidden)
    # up_proj is (intermediate, hidden), W*x gives (intermediate,)
    mul!(mlp.up_buf, mlp.up_proj, hidden)

    # gate * up
    @simd for i in 1:inter
        mlp.gate_buf[i] *= mlp.up_buf[i]
    end

    # down = down_proj(gate * up)
    # down_proj is (hidden, intermediate), W*x gives (hidden,)
    mul!(mlp.hidden_buf, mlp.down_proj, mlp.gate_buf)

    return mlp.hidden_buf
end

# --- Main forward pass ---

function forward!(model::Gemma4Model, token_ids::Vector{Int}, start_pos::Int, cache::KVCacheG4; debug::Bool=false, capture_hiddens::Union{Nothing,Vector{Vector{Float32}}}=nothing)
 config = model.config
 n = config.hidden_size
 pli_size = config.hidden_size_per_layer_input
 num_tokens = length(token_ids)
 embed_scale = config.embed_scale

 # Process each token one at a time through all layers.
 # This is the correct approach for causal autoregressive models:
 # each token attends to itself and all previous positions.

 for t in 1:num_tokens
 tid = token_ids[t]
 curr_pos = start_pos + t - 1

 # Token embedding (scaled by sqrt(hidden_size))
 embed_row = view(model.embed_tokens, tid, :) # already 1-indexed from tokenizer
 @simd for i in 1:n
 model.hidden_buf[i] = embed_row[i] * embed_scale
 end

 # Per-layer inputs for this token
 if pli_size > 0
 compute_per_layer_inputs!(model, tid, model.hidden_buf)
 end

 hidden = model.hidden_buf

 # Capture embedding output (hidden state 0) for the last token
 if capture_hiddens !== nothing && t == num_tokens
 push!(capture_hiddens, copy(hidden))
 end

 for (layer_idx, layer) in enumerate(model.layers)
 layer_idx_0 = layer_idx - 1

 # === Attention block ===
 copyto!(model.residual_buf, hidden)
 rmsnorm!(hidden, hidden, layer.input_norm_w, config.rms_norm_eps)

 attention_forward!(layer.attn, hidden, curr_pos, cache, layer_idx,
 model.sliding_cos, model.sliding_sin,
 model.full_cos, model.full_sin,
 model.shared_kv_k, model.shared_kv_v, config)

 rmsnorm!(hidden, hidden, layer.post_attn_norm_w, config.rms_norm_eps)
 @simd for i in 1:n
 hidden[i] += model.residual_buf[i]
 end

 # === MLP block ===
 copyto!(model.residual_buf, hidden)
 rmsnorm!(hidden, hidden, layer.pre_ff_norm_w, config.rms_norm_eps)
 mlp_out = mlp_forward!(layer, hidden, config)
 rmsnorm!(mlp_out, mlp_out, layer.post_ff_norm_w, config.rms_norm_eps)
 @simd for i in 1:n
 hidden[i] = model.residual_buf[i] + mlp_out[i]
 end

 # === Per-layer input block ===
 if layer.pli !== nothing && pli_size > 0
 copyto!(model.residual_buf, hidden)
 offset = (layer_idx - 1) * pli_size + 1
 pli_slice = view(model.pli_embed_buf, offset:offset+pli_size-1)
 per_layer_input_forward!(layer, pli_slice, hidden, config)
 @simd for i in 1:n
 hidden[i] = model.residual_buf[i] + layer.pli_out_buf[i]
 end
 end

 # Layer scalar
 @simd for i in 1:n
 hidden[i] *= layer.layer_scalar
 end

 # Debug: print norm after each layer for the LAST token
 if debug && t == num_tokens
 hnorm = sqrt(sum(abs2, hidden))
 @debug "Layer $(layer_idx-1) norm" hnorm
 println(" L$(lpad(layer_idx-1,2)) norm=$(round(hnorm, digits=2))")
 end

 # Capture hidden state after each layer for the last token
 if capture_hiddens !== nothing && t == num_tokens
 push!(capture_hiddens, copy(hidden))
 end

 # Store shared KV for store layers
 if !layer.attn.is_kv_shared && layer_idx_0 in config.store_kv_layers
 model.shared_kv_k[layer_idx_0] = copy(cache.k_cache[layer_idx][:, 1:cache.seq_len[layer_idx]])
 model.shared_kv_v[layer_idx_0] = copy(cache.v_cache[layer_idx][:, 1:cache.seq_len[layer_idx]])
 end
 end
 end

 # Final norm + logits (from last token's hidden state)
 rmsnorm!(model.hidden_buf, model.hidden_buf, model.final_norm_w, config.rms_norm_eps)
 return get_logits(model, model.hidden_buf)
end

function get_logits(model::Gemma4Model, hidden::Vector{Float32})
    if model.config.tie_word_embeddings
        # lm_head = embed_tokens (tied)
        # In HF: logits = lm_head(hidden) where lm_head is nn.Linear(hidden_size, vocab_size, bias=False)
        # nn.Linear computes hidden @ weight.T = (1, hidden) @ (hidden, vocab) = (1, vocab)
        # But in Julia, embed_tokens is (vocab, hidden), so:
        # embed_tokens * hidden = (vocab, hidden) * (hidden,) = (vocab,) ✓
        mul!(model.logits_buf, model.embed_tokens, hidden)
    else
        error("Non-tied lm_head not implemented for Gemma4")
    end

 # Final logit softcapping
 CommonOps.logit_softcap!(model.logits_buf, model.config.final_logit_softcapping)

    return model.logits_buf
end

# --- Generation ---

"""Simple greedy sampling with temperature and top-k."""
# Use CommonOps.softmax_sample instead of local sample_token_gemma4
const sample_token_gemma4 = CommonOps.softmax_sample

function generate_stream_gemma4(model::Gemma4Model, prompt_tokens::Vector{Int}, decode_fn::Function;
    max_tokens::Int=512,
    temperature::Float32=1.0f0,
    top_p::Float32=0.95f0,
    top_k::Int=0,
    repetition_penalty::Float32=1.0f0,
    stop_tokens::Set{Int}=Set{Int}(),
    max_context::Int=4096)

 return Channel{String}(1) do chan # buffer=1 ensures consumer sees real timing
 try
 max_cache_seq = min(model.config.max_seq_len, max_context)
 cache = Gemma4.init_kv_cache(model.config, max_cache_seq)
 # Clear shared KV state from any previous generation
 empty!(model.shared_kv_k)
 empty!(model.shared_kv_v)

 # Track token counts for repetition penalty
 token_counts = Dict{Int,Int}()
 for t in prompt_tokens
 token_counts[t] = get(token_counts, t, 0) + 1
 end

 # Process prompt tokens
 curr_pos = 0
 if !isempty(prompt_tokens)
 logits = Gemma4.forward!(model, prompt_tokens, 0, cache)
 curr_pos = length(prompt_tokens)

 # Get logits for the last prompt token
 first_logits = vec(logits) # logits_buf is already the last token's logits
 apply_repetition_penalty!(first_logits, token_counts, repetition_penalty)

 # Sample first token
 next_token = sample_token_gemma4(first_logits; temperature=temperature, top_k=top_k)

 # NOTE: curr_pos is already length(prompt_tokens), which is the correct
 # position for the first generated token. Do NOT increment here.
 token_counts[next_token] = get(token_counts, next_token, 0) + 1

 # Check stop tokens
 if next_token in stop_tokens
 return
 end

 # Decode and yield
 token_str = decode_fn([next_token])
 put!(chan, token_str)

 last_token = next_token

 # Generate remaining tokens
 tokens_generated = 1
 while tokens_generated < max_tokens
 logits = Gemma4.forward!(model, [last_token], curr_pos, cache)
 logits_vec = vec(logits)

 apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)

 next_token = sample_token_gemma4(logits_vec; temperature=temperature, top_k=top_k)

 curr_pos += 1
 token_counts[next_token] = get(token_counts, next_token, 0) + 1

 if next_token in stop_tokens
 break
 end

 token_str = decode_fn([next_token])
 put!(chan, token_str)

 last_token = next_token
 tokens_generated += 1
 end
 end
 catch e
 if !(e isa InvalidStateException)
 @error "ERROR during Gemma4 generation stream" exception=(e, catch_backtrace())
 end
 finally
 try
 close(chan)
 catch
 end
 end
 end
end

# Use CommonOps.apply_repetition_penalty! instead of local copy
const apply_repetition_penalty! = CommonOps.apply_repetition_penalty!

function generate_text_gemma4(model::Gemma4Model, tok, prompt::String;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    encode_fn::Function,
    decode_fn::Function)

    # Gemma4 chat format: <|turn>user\n prompt <turn|> <|turn>model\n
    # Token IDs (1-indexed in Julia): <|turn>=106, <turn|>=107
    # BOS = 3 (1-indexed), EOS = 2 (1-indexed)
    end_turn = 107    # <turn|> (1-indexed)

    # Build chat-formatted prompt
    chat_prompt = "<|turn>user\n$(prompt)<turn|><|turn>model\n"
    prompt_tokens = encode_fn(tok, chat_prompt)

    # Add BOS token (1-indexed: BOS=3)
    if isempty(prompt_tokens) || prompt_tokens[1] != 3
        pushfirst!(prompt_tokens, 3)  # BOS
    end

    # Stop on <turn|> (end of model turn) and EOS
    stop_tokens = Set{Int}([end_turn, 2])  # <turn|>=107, EOS=2 (1-indexed)

 stream = generate_stream_gemma4(model, prompt_tokens, decode_fn;
 max_tokens=max_tokens, temperature=temperature,
 top_k=top_k,
 repetition_penalty=repetition_penalty,
 stop_tokens=stop_tokens)

 generated_text = IOBuffer()
 t0 = time()
 token_count = 0
 for token in stream
 print(token)
 flush(stdout)
 print(generated_text, token)
 token_count += 1
 end
 if token_count > 0
 println()
 end
 elapsed = time() - t0
 tps = elapsed > 0 ? token_count / elapsed : 0.0
 println("[t/s] $(round(tps; digits=2)) tokens/s — $token_count tokens in $(round(elapsed; digits=3))s")

    return String(take!(generated_text))
end

# --- Model Display ---

function Base.show(io::IO, ::MIME"text/plain", model::Gemma4Model)
    config = model.config

    println(io, "Gemma4Model (CPU backend)")
    println(io, "├─ Model: Gemma-4")
    println(io, "├─ Hidden size: ", config.hidden_size)
    println(io, "├─ Layers: ", config.num_layers)
    println(io, "├─ Q heads: ", config.num_q_heads)
    println(io, "├─ KV heads: ", config.num_kv_heads, " (global: ", config.num_global_kv_heads, ")")
    println(io, "├─ Head dim (sliding): ", config.head_dim)
    println(io, "├─ Head dim (global): ", config.global_head_dim)
    println(io, "├─ Intermediate size: ", config.intermediate_size)
    println(io, "├─ Double-wide intermediate: ", config.double_wide_intermediate)
    println(io, "├─ Vocab size: ", config.vocab_size)
    if hasfield(typeof(config), :vocab_size_per_layer_input)
        println(io, "├─ Vocab per layer input: ", config.vocab_size_per_layer_input)
    end
    println(io, "├─ Max seq len: ", config.max_seq_len)
    println(io, "├─ Sliding window: ", config.sliding_window)
    println(io, "├─ RMS norm eps: ", config.rms_norm_eps)
    println(io, "├─ Final logit softcapping: ", config.final_logit_softcapping)
    println(io, "├─ Attention logit softcapping: ", config.attention_logits_softcapping)
    println(io, "├─ Embed scale: ", config.embed_scale)
    println(io, "├─ Per-layer input scale: ", config.per_layer_input_scale)
    println(io, "├─ Per-layer projection scale: ", config.per_layer_model_projection_scale)
    if hasfield(typeof(config), :hidden_size_per_layer_input)
        println(io, "├─ PLI size: ", config.hidden_size_per_layer_input)
    end
    if hasfield(typeof(config), :num_kv_shared_layers)
        println(io, "├─ KV shared layers: ", config.num_kv_shared_layers)
        if config.num_kv_shared_layers > 0 && hasfield(typeof(config), :first_kv_shared_layer)
            println(io, "│  └─ First shared layer index (0-based): ", config.first_kv_shared_layer)
        end
    end
    if hasfield(typeof(config), :attention_k_eq_v)
        println(io, "├─ Attention k_eq_v: ", config.attention_k_eq_v)
    end
    if hasfield(typeof(config), :sliding_rope_theta)
        println(io, "├─ Sliding RoPE theta: ", config.sliding_rope_theta)
    end
    if hasfield(typeof(config), :full_rope_theta)
        println(io, "├─ Full RoPE theta: ", config.full_rope_theta)
    end
    if hasfield(typeof(config), :full_partial_rotary_factor)
        println(io, "├─ Full partial rotary factor: ", config.full_partial_rotary_factor)
    end
    if hasfield(typeof(config), :tie_word_embeddings)
        println(io, "├─ Tie word embeddings: ", config.tie_word_embeddings)
    end

    # Count layer types if available
    if hasfield(typeof(config), :layer_types)
        sliding_count = count(==("sliding_attention"), config.layer_types)
        full_count = count(==("full_attention"), config.layer_types)
        println(io, "└─ Layer composition: ", sliding_count, " sliding, ", full_count, " full")
    end
end

function Base.show(io::IO, model::Gemma4Model)
    show(io, MIME"text/plain"(), model)
end

end # module Gemma4
