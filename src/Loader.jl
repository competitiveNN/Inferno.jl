module Loader

using ..GGUF
using ..Model
using OrderedCollections
# GPU backend (optional)
try
    using oneAPI
catch
end

export load_weights

using ..Dequant

function extract_tensor(file::GGUF.GGUFFile, info::GGUF.TensorInfo)
    num_elements = Int(prod(info.dimensions))
    start = Int(file.data_offset + info.offset) + 1

    data = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_BF16
        # BF16 to Float32 conversion (since BFloat16 is gone)
        raw_u16 = reinterpret(UInt16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
        reinterpret(Float32, UInt32.(raw_u16) .<< 16)
    elseif info.type == GGUF.GGML_TYPE_Q5_K
        dequantize_q5_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q4_K
        dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q8_0
        dequantize_q8_0(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XXS
        dims = Tuple(Int.(info.dimensions))
        inner = dims[1]
        outer = length(dims) > 1 ? dims[2] : 1
        raw_data = @view file.tensor_data[start:start+num_elements*66÷256-1]
        gpu_data = oneArray(collect(raw_data))
        return Model.IQ2XXSMatrix(gpu_data, inner, outer)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XS
        dequantize_iq2_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_S
        dequantize_iq2_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_XXS
        dequantize_iq3_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_S
        dequantize_iq3_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ4_XS
        dequantize_iq4_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ4_NL
        dequantize_iq4_nl(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q2_K
        dequantize_q2_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q3_K
        dequantize_q3_k(@view(file.tensor_data[start:end]), num_elements)
 elseif info.type == GGUF.GGML_TYPE_Q6_K
 num_blocks = num_elements ÷ 256
 data_size = num_blocks * 210
 dequantize_q6_k(collect(@view file.tensor_data[start:start+data_size-1]), num_elements)
    else
        @warn "Unhandled type, zeroing" type = info.type
        zeros(Float16, num_elements)
    end

    dims = Tuple(Int.(info.dimensions))

 if length(dims) > 2
 return reshape(data, dims)
 else
 # GGUF stores data as [inner][outer] where inner varies fastest
 # Julia is column-major, so dequantized data is in the right order
 # Just reshape to (inner, outer) without transpose
 inner = dims[1]
 outer = length(dims) > 1 ? dims[2] : 1
 return reshape(data, inner, outer)
 end
end

function extract_tensor(file::GGUF.GGUFFile, name::String)
    if !haskey(file.tensors, name)
        @warn "Tensor not found, using zero surrogate" name = name
        return zeros(Float16, 1, 1) |> oneArray
    end

    info = file.tensors[name]
    return extract_tensor(file, info)
end

function get_weight(file::GGUF.GGUFFile, info::GGUF.TensorInfo)
    tensor = extract_tensor(file, info)
    if tensor isa Model.IQ2XXSMatrix
        return tensor
    else
        # Convert to Float16 for GPU storage (most model weights are FP16)
        # Only transpose 2D matrices; multi-D tensors (e.g., mmproj) keep their shape
        if ndims(tensor) == 2
            return oneArray(Float16.(tensor'))
        else
            return oneArray(Float16.(tensor))
        end
    end
end

function get_weight(file::GGUF.GGUFFile, name::String)
    tensor = extract_tensor(file, name)
    if tensor isa Model.IQ2XXSMatrix
        return tensor
    else
        # Convert to Float16 for GPU storage (most model weights are FP16)
        # Only transpose 2D matrices; multi-D tensors (e.g., mmproj) keep their shape
        if ndims(tensor) == 2
            return oneArray(Float16.(tensor'))
        else
            return oneArray(Float16.(tensor))
        end
    end
end

function get_bias_or_norm(file::GGUF.GGUFFile, info::GGUF.TensorInfo)
    tensor = extract_tensor(file, info)
    # Convert to Float16 for GPU storage (norm weights and biases are typically FP16)
    return oneArray(Float16.(vec(collect(tensor))))
end

function get_bias_or_norm(file::GGUF.GGUFFile, name::String)
    tensor = extract_tensor(file, name)
    # Convert to Float16 for GPU storage (norm weights and biases are typically FP16)
    return oneArray(Float16.(vec(collect(tensor))))
end
struct BlockTensors
    index::Int

    # Pre/Post Normalization
    attn_norm_weight::Union{GGUF.TensorInfo,Nothing}
    input_layernorm_weight::Union{GGUF.TensorInfo,Nothing}
    post_attention_norm_weight::Union{GGUF.TensorInfo,Nothing}
    post_attention_layernorm_weight::Union{GGUF.TensorInfo,Nothing}
    ffn_norm_weight::Union{GGUF.TensorInfo,Nothing}

    # Attention (Full & MLA)
    attn_qkv_weight::Union{GGUF.TensorInfo,Nothing}
    attn_q_weight::Union{GGUF.TensorInfo,Nothing}
    attn_k_weight::Union{GGUF.TensorInfo,Nothing}
    attn_v_weight::Union{GGUF.TensorInfo,Nothing}
    attn_output_weight::Union{GGUF.TensorInfo,Nothing}
    attn_q_norm_weight::Union{GGUF.TensorInfo,Nothing}
    attn_k_norm_weight::Union{GGUF.TensorInfo,Nothing}
    attn_q_a_weight::Union{GGUF.TensorInfo,Nothing}
    attn_q_a_norm_weight::Union{GGUF.TensorInfo,Nothing}
    attn_q_b_weight::Union{GGUF.TensorInfo,Nothing}
    attn_kv_a_mqa_weight::Union{GGUF.TensorInfo,Nothing}
    attn_kv_a_norm_weight::Union{GGUF.TensorInfo,Nothing}
    attn_kv_b_weight::Union{GGUF.TensorInfo,Nothing}

    # DeltaNet (SSM)
    attn_gate_weight::Union{GGUF.TensorInfo,Nothing}
    ssm_out_weight::Union{GGUF.TensorInfo,Nothing}
    ssm_a::Union{GGUF.TensorInfo,Nothing}
    ssm_alpha_weight::Union{GGUF.TensorInfo,Nothing}
    ssm_beta_weight::Union{GGUF.TensorInfo,Nothing}
    ssm_conv1d_weight::Union{GGUF.TensorInfo,Nothing}
    ssm_dt_bias::Union{GGUF.TensorInfo,Nothing}
    ssm_norm_weight::Union{GGUF.TensorInfo,Nothing}

    # MLP
    ffn_gate_weight::Union{GGUF.TensorInfo,Nothing}
    ffn_up_weight::Union{GGUF.TensorInfo,Nothing}
    ffn_down_weight::Union{GGUF.TensorInfo,Nothing}

    expert_tensors::LittleDict{String,GGUF.TensorInfo}
end

function extract_sorted_blocks(file_tensors::Dict{String,GGUF.TensorInfo})
    temp_blocks = Dict{Int,Dict{String,GGUF.TensorInfo}}()

    for (name, tensor) in file_tensors
        m = match(r"^blk\.(\d+)\.(.*)", name)

        if m !== nothing
            blk_idx = parse(Int, m.captures[1])
            sub_name = String(m.captures[2])

            if !haskey(temp_blocks, blk_idx)
                temp_blocks[blk_idx] = Dict{String,GGUF.TensorInfo}()
            end
            temp_blocks[blk_idx][sub_name] = tensor
        end
    end
    sorted_indices = sort(collect(keys(temp_blocks)))

    result = BlockTensors[]
    for idx in sorted_indices
        tb = temp_blocks[idx]

        expert_dict = LittleDict{String,GGUF.TensorInfo}()
        for (k, v) in tb
            if occursin("exps", k)
                expert_dict[k] = v
            end
        end

        push!(result, BlockTensors(
            idx,
            get(tb, "attn_norm.weight", nothing),
            get(tb, "input_layernorm.weight", nothing),
            get(tb, "post_attention_norm.weight", nothing),
            get(tb, "post_attention_layernorm.weight", nothing),
            get(tb, "ffn_norm.weight", nothing), get(tb, "attn_qkv.weight", nothing),
            get(tb, "attn_q.weight", nothing),
            get(tb, "attn_k.weight", nothing),
            get(tb, "attn_v.weight", nothing),
            get(tb, "attn_output.weight", nothing),
            get(tb, "attn_q_norm.weight", nothing),
            get(tb, "attn_k_norm.weight", nothing),
            get(tb, "attn_q_a.weight", nothing),
            get(tb, "attn_q_a_norm.weight", nothing),
            get(tb, "attn_q_b.weight", nothing),
            get(tb, "attn_kv_a_mqa.weight", nothing),
            get(tb, "attn_kv_a_norm.weight", nothing),
            get(tb, "attn_kv_b.weight", nothing), get(tb, "attn_gate.weight", nothing),
            get(tb, "ssm_out.weight", nothing),
            get(tb, "ssm_a", nothing),
            get(tb, "ssm_alpha.weight", nothing),
            get(tb, "ssm_beta.weight", nothing),
            get(tb, "ssm_conv1d.weight", nothing),
            get(tb, "ssm_dt.bias", nothing),
            get(tb, "ssm_norm.weight", nothing), get(tb, "ffn_gate.weight", nothing),
            get(tb, "ffn_up.weight", nothing),
            get(tb, "ffn_down.weight", nothing), expert_dict
        ))
    end
    return result
end

function load_weights(file::GGUF.GGUFFile, config::Model.QwenConfig;
    mmproj::Union{GGUF.GGUFFile,Nothing}=nothing)
    arch = config.architecture
    embed_raw = extract_tensor(file, "token_embd.weight")
    # Ensure embedding is Float16 for consistency with model weights
    embed = embed_raw isa Model.IQ2XXSMatrix ? embed_raw : Float16.(embed_raw)
    T_model = eltype(embed) # Use native type of embedding as model baseline

    layers = Model.DecoderLayer[]
    blocks = extract_sorted_blocks(file.tensors)

    for block in blocks
        print(".")
        flush(stdout)

        i = block.index
        is_ssm = block.ssm_a !== nothing

        in_norm_info = block.attn_norm_weight !== nothing ? block.attn_norm_weight :
                       block.input_layernorm_weight !== nothing ? block.input_layernorm_weight :
                       block.attn_norm_weight

        in_norm_w = get_bias_or_norm(file, in_norm_info)
        in_norm = Model.RMSNorm(in_norm_w, config.rms_norm_eps)

        post_norm_info = block.post_attention_norm_weight !== nothing ? block.post_attention_norm_weight :
                         block.ffn_norm_weight !== nothing ? block.ffn_norm_weight :
                         block.post_attention_layernorm_weight !== nothing ? block.post_attention_layernorm_weight :
                         block.attn_norm_weight
        post_norm_w = get_bias_or_norm(file, post_norm_info)
        post_norm = Model.RMSNorm(post_norm_w, config.rms_norm_eps)

        op = if is_ssm
            in_proj = get_weight(file, block.attn_qkv_weight)
            gate_proj = get_weight(file, block.attn_gate_weight)
            ssm_out = get_weight(file, block.ssm_out_weight)

            ssm_a = get_bias_or_norm(file, block.ssm_a)
            ssm_alpha = get_weight(file, block.ssm_alpha_weight)
            ssm_beta = get_weight(file, block.ssm_beta_weight)
            ssm_conv1d_raw = extract_tensor(file, block.ssm_conv1d_weight)
            ssm_conv1d_gpu = oneArray(Float16.(ssm_conv1d_raw))
            oneAPI.synchronize()
            ssm_dt_bias = get_bias_or_norm(file, block.ssm_dt_bias)
            ssm_norm_w = get_bias_or_norm(file, block.ssm_norm_weight)
            ssm_norm = Model.RMSNorm(ssm_norm_w, config.rms_norm_eps)

            conv_kernel = config.ssm_conv_kernel
            num_v_heads = config.ssm_time_step_rank
            num_k_heads = config.ssm_group_count
            head_k_dim = config.ssm_state_size
            head_v_dim = config.ssm_inner_size ÷ num_v_heads
            conv_channels = config.ssm_inner_size + 2 * num_k_heads * head_k_dim

            conv_state = zeros(Float32, conv_channels, conv_kernel)
            ssm_state = zeros(Float32, head_v_dim, head_k_dim, num_v_heads)
            ssm_conv1d_cpu = collect(Float32.(ssm_conv1d_gpu))

            Model.GatedDeltaNet(i, in_proj, gate_proj, ssm_out,
                ssm_conv1d_gpu, ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias, ssm_norm, config)
        elseif arch == :deepseek2
            q_a_proj = get_weight(file, block.attn_q_a_weight)
            q_a_norm = Model.RMSNorm(get_bias_or_norm(file, block.attn_q_a_norm_weight), config.rms_norm_eps)
            q_b_proj = get_weight(file, block.attn_q_b_weight)
            kv_a_proj_with_mqa = get_weight(file, block.attn_kv_a_mqa_weight)
            kv_a_norm = Model.RMSNorm(get_bias_or_norm(file, block.attn_kv_a_norm_weight), config.rms_norm_eps)
            kv_b_proj = get_weight(file, block.attn_kv_b_weight)
            wo = get_weight(file, block.attn_output_weight)

            Model.MLAttention(q_a_proj, q_a_norm, q_b_proj, kv_a_proj_with_mqa, kv_a_norm, kv_b_proj, wo,
                config.num_attention_heads, config.head_dim, config.q_lora_rank, config.kv_lora_rank,
                config.qk_rope_head_dim, config.v_head_dim, config.qk_nope_head_dim,
                T_model(1.0) / sqrt(T_model(config.head_dim)), config)
        else
            wqkv = arch == :phi3 && block.attn_qkv_weight !== nothing ? get_weight(file, block.attn_qkv_weight) : nothing
            qw = arch != :phi3 && block.attn_q_weight !== nothing ? get_weight(file, block.attn_q_weight) : nothing
            kw = arch != :phi3 && block.attn_k_weight !== nothing ? get_weight(file, block.attn_k_weight) : nothing
            vw = arch != :phi3 && block.attn_v_weight !== nothing ? get_weight(file, block.attn_v_weight) : nothing
            ow = get_weight(file, block.attn_output_weight)

            q_norm_w = block.attn_q_norm_weight !== nothing ? get_bias_or_norm(file, block.attn_q_norm_weight) : nothing
            k_norm_w = block.attn_k_norm_weight !== nothing ? get_bias_or_norm(file, block.attn_k_norm_weight) : nothing
            q_norm = isnothing(q_norm_w) ? nothing : Model.RMSNorm(q_norm_w, config.rms_norm_eps)
            k_norm = isnothing(k_norm_w) ? nothing : Model.RMSNorm(k_norm_w, config.rms_norm_eps)

            n_heads = config.num_attention_heads
            n_kv = config.num_key_value_heads

            Model.FullAttention(i, arch, qw, kw, vw, wqkv, ow, q_norm, k_norm, n_heads, n_kv, config.head_dim, config)
        end

        mlp = if haskey(block.expert_tensors, "ffn_gate_exps.0.weight") || haskey(block.expert_tensors, "ffn_gate_exps.weight")
            gate = get_weight(file, block.ffn_gate_weight)
            n_exp = config.num_experts
            experts_gate = [get_weight(file, block.expert_tensors["ffn_gate_exps.$(j-1).weight"]) for j in 1:n_exp]
            experts_up = [get_weight(file, block.expert_tensors["ffn_up_exps.$(j-1).weight"]) for j in 1:n_exp]
            experts_down = [get_weight(file, block.expert_tensors["ffn_down_exps.$(j-1).weight"]) for j in 1:n_exp]
            Model.MoE(i, gate, experts_gate, experts_up, experts_down, n_exp, config.num_experts_per_tok)
        else
            gate_w = get_weight(file, block.ffn_gate_weight)
            up_w = get_weight(file, block.ffn_up_weight)
            down_w = get_weight(file, block.ffn_down_weight)
            Model.MLP(i, gate_w, up_w, down_w)
        end

        push!(layers, Model.DecoderLayer(in_norm, op, post_norm, mlp, is_ssm))
    end

    final_norm_w = get_bias_or_norm(file, "output_norm.weight")
    final_norm = Model.RMSNorm(final_norm_w, config.rms_norm_eps)

    lm_head_raw = haskey(file.tensors, "output.weight") ?
                  extract_tensor(file, "output.weight") : embed
    lm_head = lm_head_raw # Use native type

    rope_dim = Int(get(file.metadata, "$(arch).rope.dimension_count", config.head_dim))
    partial_rotary_factor = Float32(get(file.metadata, "$(arch).partial_rotary_factor", 1.0))
    rotary_dim = round(Int, rope_dim * partial_rotary_factor)
    rope = Model.RotaryEmbedding(rope_dim; base=config.rope_theta, rotary_dim=rotary_dim)

    mmproj_data = nothing
    if mmproj !== nothing
        @info "Integrating mmproj tensors"
        mmproj_data = Dict{String,Any}()
        for name in keys(mmproj.tensors)
            mmproj_data[name] = get_weight(mmproj, name)
        end
    end

    return Model.QwenModel(config, embed, layers, final_norm, lm_head, rope, mmproj_data)
end

end # module
