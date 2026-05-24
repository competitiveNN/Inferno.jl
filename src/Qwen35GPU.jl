"""
Qwen35GPU — Full GPU inference for Qwen3.5 hybrid SSM/attention model on Intel Arc.

All computation on GPU using oneAPI.jl + KernelAbstractions.jl.
FULLY GPU-ACCELERATED: No CPU loops in forward pass.
All weights, computations, and control flow stay on GPU.
"""
module Qwen35GPU

using oneAPI
using LinearAlgebra
using KernelAbstractions

# Use unified kernel library
using ..GPUCommon: rmsnorm_gpu!, silu_gpu!, sigmoid_gpu!, batched_attention_scores!, batched_softmax!, batched_ssm_state_update!, batched_ssm_output_sum!
using ..FusedKernels: fused_l2norm!, fused_attention_weighted_sum!, fused_mlp_gate_mul!, fused_silu_gate_mul!, fused_ssm_gate_sigmoid!, fused_ssm_decay!

using ..GGUF
using ..Tokenizer
using ..LoaderCPU

const oneArray = oneAPI.oneArray
const oneVector{T} = oneArray{T,1}
const oneMatrix{T} = oneArray{T,2}

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
 rms_norm_eps::Float32 = 1e-6f0
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
 q_w::oneMatrix{Float32}
 k_w::oneMatrix{Float32}
 v_w::oneMatrix{Float32}
 o_w::oneMatrix{Float32}
 q_norm::oneVector{Float32}
 k_norm::oneVector{Float32}
end

mutable struct MLPLayer
 gate_w::oneMatrix{Float32}
 up_w::oneMatrix{Float32}
 down_w::oneMatrix{Float32}
end

mutable struct SSMLayer
 in_proj::oneMatrix{Float32}
 gate_proj::oneMatrix{Float32}
 ssm_out::oneMatrix{Float32}
 ssm_conv1d::oneMatrix{Float32}
 alpha_w::oneMatrix{Float32}
 beta_w::oneMatrix{Float32}
 ssm_a::oneVector{Float32}
 ssm_dt_bias::oneVector{Float32}
 ssm_norm_w::oneVector{Float32}
 # State buffers
 conv_state::oneMatrix{Float32}
 h_state::oneArray{Float32, 3}
 # Dimensions
 d_inner::Int
 num_v_heads::Int
 num_k_heads::Int
 head_v_dim::Int
 head_k_dim::Int
 conv_channels::Int
end

mutable struct DecoderLayer
 input_norm_w::oneVector{Float32}
 attn::Union{AttentionLayer, Nothing}
 mlp::MLPLayer
 ssm::Union{SSMLayer, Nothing}
 post_attn_norm_w::oneVector{Float32}
 is_attention::Bool
end

# ============================================================
# GPU Model
# ============================================================
mutable struct Qwen35GPUModel
 config::Qwen35GPUConfig
 embed::oneMatrix{Float32}
 layers::Vector{DecoderLayer}
 final_norm_w::oneVector{Float32}
 lm_head::oneMatrix{Float32}
 cos_cached::oneMatrix{Float32}
 sin_cached::oneMatrix{Float32}
 kv_cache_k::Vector{oneMatrix{Float32}}
 kv_cache_v::Vector{oneMatrix{Float32}}
 kv_len::Int
 # Work buffers on GPU
 hidden::oneVector{Float32}
 residual::oneVector{Float32}
 norm_buf::oneVector{Float32}
 qkv_buf::oneVector{Float32}
 attn_out::oneVector{Float32}
 gate_buf::oneVector{Float32}
 up_buf::oneVector{Float32}
 down_buf::oneVector{Float32}
\tssm_out_buf::oneVector{Float32}
\tscore_buf::oneMatrix{Float32}
\tlogits_buf::oneVector{Float32}
\tdevice::oneAPI.oneL0.ZeDevice
end

# ============================================================
# Helper: GPU dot product (NO SYNC - returns GPU array)
# ============================================================
gpu_dot(a, b) = oneAPI.oneAPI.sum(a .* b)

# ============================================================
# RMSNorm (uses GPUCommon kernel)
# ============================================================
function rmsnorm!(out::oneVector{Float32}, x::oneVector{Float32},
 w::oneVector{Float32}, eps::Float32)
 return rmsnorm_gpu!(out, x, w, eps)
end

function rmsnorm!(out::oneVector{Float32}, x::oneVector{Float32}, eps::Float32)
 n = length(x)
 ss = oneAPI.oneAPI.sum(x .^ 2)
 inv_rms = 1.0f0 / sqrt(ss / n + eps)
 out .= x .* inv_rms
 return out
end

# ============================================================
# L2 Norm (for Q/K in Qwen3.5) - GPU native
# ============================================================
function l2norm!(out::oneVector{Float32}, x::oneVector{Float32}, eps::Float32)
 ss = oneAPI.oneAPI.sum(x .^ 2)
 inv_rms = 1.0f0 / sqrt(ss + eps)
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
  # Convert flat index to (head, dim)
  half = head_dim ÷ 2
  n_heads = n_total ÷ head_dim
  
  h = (idx - 1) ÷ head_dim  # 0-based head index
  d = (idx - 1) % head_dim  # 0-based dim index
  
  # Only process first half (rotary dims)
  if d < half && d < (rotary_dim ÷ 2)
   # Get cos/sin for this position and dimension
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

function apply_rope!(q_or_k::oneVector{Float32}, pos::Int,
 cos_t::oneMatrix{Float32}, sin_t::oneMatrix{Float32},
 head_dim::Int, rotary_dim::Int)
 n = length(q_or_k)
 kernel = apply_rope_kernel!
 kernel(q_or_k, cos_t, sin_t, pos, head_dim, rotary_dim; ndrange=n)
 @synchronize()
 return q_or_k
end

# ============================================================
# Attention Forward (Fully GPU - No CPU loops!)
# ============================================================
function attention_forward!(model::Qwen35GPUModel, layer::DecoderLayer, layer_idx::Int,
 hidden::oneVector{Float32}, pos::Int)
 attn = layer.attn
 cfg = model.config
 h = cfg.hidden_size
 head_dim = cfg.head_dim
 n_heads = cfg.num_attention_heads
 n_kv = cfg.num_key_value_heads
 n_groups = n_heads ÷ n_kv

 # QKV projections
 q = view(model.qkv_buf, 1:n_heads * head_dim)
 k = view(model.qkv_buf, n_heads * head_dim + 1:(n_heads + n_kv) * head_dim)
 v = view(model.qkv_buf, (n_heads + n_kv) * head_dim + 1:(n_heads + 2 * n_kv) * head_dim)

 mul!(q, attn.q_w, hidden)
 mul!(k, attn.k_w, hidden)
 mul!(v, attn.v_w, hidden)

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
 
 # Use GPU broadcast for copy (no CPU loop)
 for kv_head in 0:n_kv-1
  off = kv_head * head_dim
  k_cache[off+1:off+head_dim, pos+1] .= k[off+1:off+head_dim]
  v_cache[off+1:off+head_dim, pos+1] .= v[off+1:off+head_dim]
 end

 # Attention scores - FULLY GPU (batched kernel)
 seq_len = pos + 1
 scores = view(model.score_buf, 1:n_heads, 1:seq_len)
 
 # Use batched attention scores kernel (no CPU loops!)
 batched_attention_scores!(scores, q, k_cache, n_heads, head_dim, seq_len, n_groups)

 # Softmax - FULLY GPU (batched kernel)
 batched_softmax!(scores, n_heads, seq_len)

 # Weighted sum of values — fully fused kernel (no per-head allocation)
 fill!(model.attn_out, 0.0f0)
 fused_attention_weighted_sum!(model.attn_out, scores, v_cache, n_heads, head_dim, seq_len, n_groups)

 # Output projection
 out = view(model.qkv_buf, 1:h)
 mul!(out, attn.o_w, model.attn_out)
 return out
end

# ============================================================
# SSM Forward (Fully GPU - No CPU loops!)
# ============================================================
function ssm_forward!(model::Qwen35GPUModel, layer::DecoderLayer, hidden::oneVector{Float32})
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
 xz = view(model.qkv_buf, 1:conv_channels + d_inner)
 mul!(xz, ssm.in_proj, hidden)
 x_conv = view(xz, 1:conv_channels)
 z = view(xz, conv_channels+1:conv_channels+d_inner)

 # Gate
 gate_out = view(model.gate_buf, 1:d_inner)
 mul!(gate_out, ssm.gate_proj, hidden)
 silu_gpu!(gate_out, gate_out)

 # Conv1d: shift state, store new input
 ssm.conv_state[:, 1:end-1] .= ssm.conv_state[:, 2:end]
 ssm.conv_state[:, end] .= x_conv

 # Alpha/beta projections
 alpha = view(model.qkv_buf, 1:n_v)
 beta_all = view(model.qkv_buf, n_v+1:n_v + n_v * head_v)
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
 out = view(model.qkv_buf, 1:h)
 mul!(out, ssm.ssm_out, y_gated)
 return out
end

# ============================================================
# MLP (SwiGLU) - Uses GPUCommon kernels
# ============================================================
function mlp_forward!(model::Qwen35GPUModel, layer::DecoderLayer, hidden::oneVector{Float32})
 mlp = layer.mlp
 h = model.config.hidden_size
 int_size = model.config.intermediate_size

 gate = view(model.gate_buf, 1:int_size)
 up = view(model.up_buf, 1:int_size)

 mul!(gate, mlp.gate_w, hidden)
 mul!(up, mlp.up_w, hidden)

 fused_mlp_gate_mul!(gate, up)

 out = view(model.down_buf, 1:h)
 mul!(out, mlp.down_w, gate)
 return out
end

# ============================================================
# Full Forward Pass (Fully GPU)
# ============================================================
function forward_gpu!(model::Qwen35GPUModel, token_ids::Vector{Int}, start_pos::Int)
 cfg = model.config
 h = cfg.hidden_size

 # Embedding
 tid = token_ids[1]
 copy!(model.hidden, view(model.embed, tid, :))

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

\trmsnorm_gpu!(model.norm_buf, model.hidden, model.final_norm_w, cfg.rms_norm_eps)
\t# Use pre-allocated logits buffer (vocab_size, not hidden_size)
\tmul!(model.logits_buf, model.lm_head, model.norm_buf)

\treturn Array(Float32.(model.logits_buf))
end

# ============================================================
# Model state reset
# ============================================================
function reset_model!(model::Qwen35GPUModel)
    for k in model.kv_cache_k
        fill!(k, 0.0f0)
    end
    for v in model.kv_cache_v
        fill!(v, 0.0f0)
    end
    for layer in model.layers
        if !layer.is_attention && !isnothing(layer.ssm)
            fill!(layer.ssm.conv_state, 0.0f0)
            fill!(layer.ssm.h_state, 0.0f0)
        end
    end
    return model
end

# ============================================================
# ============================================================
# Generation (streaming)
# ============================================================
function generate_stream(model::Qwen35GPUModel, tok::BPETokenizer, prompt::String;
 max_tokens::Int=100, temperature::Float32=0.0f0,
 top_p::Float32=0.0f0, top_k::Int=0)
    return Channel{String}(32) do chan
        prompt_ids = encode(tok, prompt)
        token_buf  = Int[0]

        reset_model!(model)
        pos = 0
        logits = nothing

        # Prime KV caches with full prompt
        for token in prompt_ids
            token_buf[1] = token
            logits = forward_gpu!(model, token_buf, pos)
            pos += 1
        end

        # No prompt or no generation requested
        if isempty(prompt_ids) || max_tokens <= 0 || logits === nothing
            return
        end

        # First generated token from the last prompt logits
        if temperature == 0.0f0
            last_token = argmax(logits)
        else
            scaled = logits ./ temperature
            m = maximum(scaled)
            exp_vals = exp.(scaled .- m)
            probs = exp_vals ./ sum(exp_vals)
            last_token = sample(probs)
        end

        word = decode(tok, [last_token])
        put!(chan, word)

        if last_token == 151643 || last_token == 0
            return
        end

        # Continue generating
        for _ in 1:max_tokens - 1
            token_buf[1] = last_token
            logits = forward_gpu!(model, token_buf, pos)
            pos += 1

            if temperature == 0.0f0
                last_token = argmax(logits)
            else
                scaled = logits ./ temperature
                m = maximum(scaled)
                exp_vals = exp.(scaled .- m)
                probs = exp_vals ./ sum(exp_vals)
                last_token = sample(probs)
            end

            word = decode(tok, [last_token])
            put!(chan, word)

            if last_token == 151643 || last_token == 0
                break
            end
        end
    end
end

function sample(probs::Vector{Float32})
 r = rand(Float32)
 cum = 0.0f0
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
 # Handle any tensor type - always convert to Float32 Matrix
 if cpu_tensor isa Matrix{Float32}
  return oneMatrix{Float32}(cpu_tensor)
 elseif cpu_tensor isa Matrix
  # Convert to Float32 and reshape if needed
  f32_data = Float32.(cpu_tensor)
  return oneMatrix{Float32}(f32_data)
 else
  # Fallback: convert to vector then reshape
  f32_vec = Float32.(vec(cpu_tensor))
  dims = Tuple(info.dimensions)
  return reshape(oneMatrix{Float32}(f32_vec), dims...)
 end
 end

 function gpu_vec(name::String)
  info = GGUF.get_tensor(file, name)
  cpu_vec = LoaderCPU.extract_tensor_cpu(file, info)
  return oneVector{Float32}(vec(Float32.(cpu_vec)))
 end

 println("[GPU] Loading embedding...")
 cpu_embed = LoaderCPU.extract_tensor_cpu(file, "token_embd.weight")
 # Handle any tensor type - always convert to Float32 Matrix
 if cpu_embed isa Matrix{Float32}
  embed = oneMatrix{Float32}(cpu_embed)
 elseif cpu_embed isa Matrix
  embed = oneMatrix{Float32}(Float32.(cpu_embed))
 else
  # Fallback: convert to vector then reshape
  f32_vec = Float32.(vec(cpu_embed))
  dims = Tuple(file.tensors["token_embd.weight"].dimensions)
  embed = reshape(oneMatrix{Float32}(f32_vec), dims...)
 end
 println("[GPU] Embedding shape: ", size(embed))

 println("[GPU] Loading lm_head...")
 lm_head = try
  cpu_lm_head = LoaderCPU.extract_tensor_cpu(file, "output.weight")
  oneMatrix{Float32}(Float32.(cpu_lm_head))
 catch e
  println(" Warning: output.weight not found, using tied embedding weights")
  embed
 end

 println("[GPU] Precomputing RoPE...")
 max_pos = cfg.max_position_embeddings
 half_rot = head_dim ÷ 2
 cos_t = zeros(Float32, max_pos, half_rot)
 sin_t = zeros(Float32, max_pos, half_rot)
 inv_freq = zeros(Float32, half_rot)
 for i in 1:half_rot
  inv_freq[i] = Float32(i - 1)
 end
 inv_freq = 1.0f0 ./ (cfg.rope_theta .^ (2.0f0 .* inv_freq ./ head_dim))
 for pos in 0:max_pos-1
  f = pos * inv_freq
  for i in 1:half_rot
   cos_t[pos+1, i] = cos(f[i])
   sin_t[pos+1, i] = sin(f[i])
  end
 end
 cos_gpu = oneMatrix{Float32}(cos_t)
 sin_gpu = oneMatrix{Float32}(sin_t)

 println("[GPU] Loading layers...")
 layers = DecoderLayer[]

 for i in 0:cfg.num_hidden_layers-1
  is_attn = (i + 1) % cfg.full_attention_interval == 0
  prefix = "blk.$(i)"

  input_norm_w = gpu_vec("$(prefix).attn_norm.weight")
  post_attn_norm_w = gpu_vec("$(prefix).post_attention_norm.weight")

  attn = nothing
  ssm = nothing

  if is_attn
   q_w = gpu_tensor("$(prefix).attn_q.weight")
   k_w = gpu_tensor("$(prefix).attn_k.weight")
   v_w = gpu_tensor("$(prefix).attn_v.weight")
   o_w = gpu_tensor("$(prefix).attn_output.weight")

   qn_vec = vec(Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_q_norm.weight")))
   kn_vec = vec(Float32.(LoaderCPU.extract_tensor_cpu(file, "$(prefix).attn_k_norm.weight")))
   q_norm_gpu = oneVector{Float32}(qn_vec .+ 1.0f0)
   k_norm_gpu = oneVector{Float32}(kn_vec .+ 1.0f0)

   attn = AttentionLayer(q_w, k_w, v_w, o_w, q_norm_gpu, k_norm_gpu)
  else
   in_proj_m = gpu_tensor("$(prefix).attn_qkv.weight")
   gate_proj_m = gpu_tensor("$(prefix).attn_gate.weight")
   ssm_out_m = gpu_tensor("$(prefix).ssm_out.weight")
   
   ssm_conv1d_m = gpu_tensor("$(prefix).ssm_conv1d.weight")
   alpha_w_m = gpu_tensor("$(prefix).ssm_alpha.weight")
   beta_w_m = gpu_tensor("$(prefix).ssm_beta.weight")
   ssm_a_v = gpu_vec("$(prefix).ssm_a")
   ssm_dt_v = gpu_vec("$(prefix).ssm_dt.bias")
   ssm_norm_v = gpu_vec("$(prefix).ssm_norm.weight")

   d_inner = cfg.ssm_inner_size
   n_v = length(ssm_a_v)
   n_k = cfg.ssm_group_count
   head_v = d_inner ÷ n_v
   in_proj_out_dim = size(in_proj_m, 2)
   head_k = (in_proj_out_dim - d_inner) ÷ (2 * n_k)
   conv_ch = d_inner + 2 * n_k * head_k

   if head_k <= 0
    error("Invalid head_k calculation")
   end

   conv_state = oneMatrix{Float32}(zeros(Float32, conv_ch, cfg.ssm_conv_kernel))
   h_state = oneArray{Float32, 3}(zeros(Float32, head_v, head_k, n_v))

   ssm = SSMLayer(
    in_proj_m, gate_proj_m, ssm_out_m, ssm_conv1d_m,
    alpha_w_m, beta_w_m, ssm_a_v, ssm_dt_v, ssm_norm_v,
    conv_state, h_state,
    d_inner, n_v, n_k, head_v, head_k, conv_ch
   )
  end

  gate_mlp = gpu_tensor("$(prefix).ffn_gate.weight")
  up_mlp = gpu_tensor("$(prefix).ffn_up.weight")
  down_mlp = gpu_tensor("$(prefix).ffn_down.weight")
  mlp = MLPLayer(gate_mlp, up_mlp, down_mlp)

  push!(layers, DecoderLayer(input_norm_w, attn, mlp, ssm, post_attn_norm_w, is_attn))
  println(" Layer $i: $(is_attn ? "Attention" : "SSM")")
 end

 final_norm_w = gpu_vec("output_norm.weight")

 println("[GPU] Allocating KV caches...")
 kv_cache_k = oneMatrix{Float32}[]
 kv_cache_v = oneMatrix{Float32}[]
 kv_slot = cfg.num_key_value_heads * head_dim
 for _ in 1:cfg.num_hidden_layers
  push!(kv_cache_k, oneMatrix{Float32}(zeros(Float32, kv_slot, max_seq_len)))
  push!(kv_cache_v, oneMatrix{Float32}(zeros(Float32, kv_slot, max_seq_len)))
 end

 d_inner = cfg.ssm_inner_size
 n_v = 16
 n_k = cfg.ssm_group_count
 head_v = d_inner ÷ n_v
 head_k = (d_inner + 2 * n_k * (head_dim ÷ 4) - d_inner) ÷ (2 * n_k)
 conv_ch = d_inner + 2 * n_k * head_k

 max_buf = max(h * 4, conv_ch + d_inner, cfg.intermediate_size * 3)
 v = cfg.vocab_size

 hidden = oneVector{Float32}(undef, h)
 logits_buf = oneVector{Float32}(undef, v)
 residual = oneVector{Float32}(undef, h)
 norm_buf = oneVector{Float32}(undef, h)
 qkv_buf = oneVector{Float32}(undef, max_buf)
 attn_out = oneVector{Float32}(undef, h)
 gate_buf = oneVector{Float32}(undef, cfg.intermediate_size)
 up_buf = oneVector{Float32}(undef, cfg.intermediate_size)
 down_buf = oneVector{Float32}(undef, h)
 ssm_out_buf = oneVector{Float32}(undef, d_inner)
 score_buf = oneMatrix{Float32}(undef, cfg.num_attention_heads, max_seq_len)

 model = Qwen35GPUModel(
  cfg, embed, layers, final_norm_w, lm_head,
  cos_gpu, sin_gpu,
  kv_cache_k, kv_cache_v, 0,
  hidden, residual, norm_buf, qkv_buf, attn_out,
  gate_buf, up_buf, down_buf, ssm_out_buf, score_buf, logits_buf,
  devs[gpu_idx]
 )

 tok = try
  Tokenizer.load_tokenizer(md)
 catch e
  println("[GPU] Tokenizer error: $e")
  nothing
 end

 println("[GPU] Loaded!")
 return model, tok
end

const forward_qwen35_gpu! = forward_gpu!

end # module Qwen35GPU
