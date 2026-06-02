"""CPU-only model loader for Inferno.jl
Loads GGUF models without GPU dependencies.
"""
module LoaderCPU

using ..ModelCPU
using ..GGUF
using ..Dequant
using ..QuantsCPU
using ..QuantizedKernels
using ..Tokenizer
using ..Safetensors: load_safetensors_model
using ..ArrowLake
using ..Jamba
using BFloat16s

export load_model_cpu, detect_model_format, extract_tensor_cpu

# Union type for weight matrices (Float32, BFloat16, or quantized)
const WeightMatrix = Union{Matrix{Float32}, Matrix{BFloat16}, Q4_K_Matrix, Q5_K_Matrix, Q6_K_Matrix, Q8_0_Matrix}

"""
    maybe_bf16(weight, use_bf16) -> Union{Matrix{Float32}, Matrix{BFloat16}}

Convert a Float32 weight matrix to BFloat16 if requested.
Only use for memory-bound matmuls (lm_head) - compute-bound matmuls 
(SSM/MLP) are faster with BLAS F32 than BF16 software conversion.
"""
function maybe_bf16(weight::Matrix{Float32}, use_bf16::Bool)
    if use_bf16 && ArrowLake.has_arrow_lake_features()
        return ArrowLake.to_bfloat16_weights(weight)
    end
    return weight
end

function maybe_bf16(weight, use_bf16::Bool)
    # Quantized weights - pass through
    return weight
end

"""
 weight_outer_dim(w) -> Int

Get the outer dimension of a weight matrix (works for both Float32 and quantized matrices).
This is the number of rows in the matrix after GGUF transpose.
"""
function weight_outer_dim end

"""
 detect_model_format(path) -> Symbol

Detect whether the path points to a GGUF file, safetensors directory, or safetensors file.
Returns :gguf, :safetensors, or :unknown.
"""
function detect_model_format(path::String)
 if isfile(path)
 # It's a file - check extension
 _, ext = splitext(path)
 if lowercase(ext) == ".gguf"
 return :gguf
 elseif occursin("safetensors", lowercase(path))
 return :safetensors
 else
 return :unknown
 end
 elseif isdir(path)
 # It's a directory - check for safetensors files or GGUF files
 files = readdir(path)
 safetensors_files = filter(f -> occursin("safetensors", lowercase(f)) && endswith(lowercase(f), ".safetensors"), files)
 gguf_files = filter(f -> endswith(lowercase(f), ".gguf"), files)
 
 if !isempty(safetensors_files)
 return :safetensors
 elseif !isempty(gguf_files)
 return :gguf
 else
 # Check for config.json (indicates HuggingFace format)
 if any(f -> lowercase(f) == "config.json", files)
 return :safetensors
 end
 return :unknown
 end
 else
 return :unknown
 end
end

# Helper to get outer dimension of weight matrix (works for Float32, BFloat16, and quantized)
weight_outer_dim(w::Matrix{Float32}) = size(w, 1)
weight_outer_dim(w::Matrix{BFloat16}) = size(w, 1)
weight_outer_dim(w::Q4_K_Matrix) = w.outer_dim
weight_outer_dim(w::Q5_K_Matrix) = w.outer_dim
weight_outer_dim(w::Q6_K_Matrix) = w.outer_dim
weight_outer_dim(w::Q8_0_Matrix) = w.outer_dim

weight_inner_dim(w::Matrix{Float32}) = size(w, 2)
weight_inner_dim(w::Matrix{BFloat16}) = size(w, 2)
weight_inner_dim(w::Q4_K_Matrix) = w.inner_dim
weight_inner_dim(w::Q5_K_Matrix) = w.inner_dim
weight_inner_dim(w::Q6_K_Matrix) = w.inner_dim
weight_inner_dim(w::Q8_0_Matrix) = w.inner_dim

"""
 extract_tensor_cpu(file::GGUF.GGUFFile, info::GGUF.TensorInfo; keep_quantized=false) -> Union{Matrix, QuantizedMatrix}

Extract tensor data from a GGUF file, with optional quantized preservation.

# Arguments
- `file`: The GGUFFile to extract from
- `info`: TensorInfo containing tensor metadata
- `keep_quantized::Bool`: If true, return quantized matrix wrapper; if false, dequantize to Float32

# Returns
Either a `Matrix{Float32}` or a quantized matrix wrapper (Q4_K_Matrix, Q5_K_Matrix, etc.)

# Supported Types
- F32, F16, BF16: Direct extraction
- Q4_K, Q5_K, Q6_K, Q8_0: Quantized extraction (with `keep_quantized=true`)
- IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS: Dequantized to Float32
"""
function extract_tensor_cpu(file::GGUF.GGUFFile, info::GGUF.TensorInfo; keep_quantized::Bool=false)
 num_elements = Int(prod(info.dimensions))
 # NOTE: file.tensor_data starts at byte file.data_offset in the GGUF file
 # The tensor offset is relative to the data section start
 # So: tensor_data[info.offset+1] = file byte file.data_offset + info.offset
 start = Int(file.data_offset + info.offset) + 1
    
    dims = Tuple(Int.(info.dimensions))
    inner = dims[1]
    outer = length(dims) > 1 ? dims[2] : 1

    if keep_quantized
        # Return quantized matrix wrapper instead of dequantizing
        if info.type == GGUF.GGML_TYPE_Q4_K
            num_blocks = num_elements ÷ 256
            data_size = num_blocks * QuantsCPU.Q4_K_BLOCK_SIZE
            data = collect(@view file.tensor_data[start:start+data_size-1])
            return Q4_K_Matrix(data, inner, outer)
        elseif info.type == GGUF.GGML_TYPE_Q5_K
            num_blocks = num_elements ÷ 256
            data_size = num_blocks * QuantsCPU.Q5_K_BLOCK_SIZE
            data = collect(@view file.tensor_data[start:start+data_size-1])
            return Q5_K_Matrix(data, inner, outer)
        elseif info.type == GGUF.GGML_TYPE_Q6_K
            num_blocks = num_elements ÷ 256
            data_size = num_blocks * QuantsCPU.Q6_K_BLOCK_SIZE
            data = collect(@view file.tensor_data[start:start+data_size-1])
            return Q6_K_Matrix(data, inner, outer)
        elseif info.type == GGUF.GGML_TYPE_Q8_0
            num_blocks = num_elements ÷ 32
            data_size = num_blocks * QuantsCPU.Q8_0_BLOCK_SIZE
            data = collect(@view file.tensor_data[start:start+data_size-1])
            return Q8_0_Matrix(data, inner, outer)
        end
        # Fall through for other types
    end

    data = if info.type == GGUF.GGML_TYPE_F32
        reinterpret(Float32, @view file.tensor_data[start:start+num_elements*4-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_F16
        reinterpret(Float16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
    elseif info.type == GGUF.GGML_TYPE_BF16
        raw_u16 = reinterpret(UInt16, @view file.tensor_data[start:start+num_elements*2-1]) |> collect
        reinterpret(Float32, UInt32.(raw_u16) .<< 16)
    elseif info.type == GGUF.GGML_TYPE_Q5_K
        Dequant.dequantize_q5_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q4_K
        Dequant.dequantize_q4_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q8_0
        Dequant.dequantize_q8_0(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_XS
        Dequant.dequantize_iq2_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ2_S
        Dequant.dequantize_iq2_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_XXS
        Dequant.dequantize_iq3_xxs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ3_S
        Dequant.dequantize_iq3_s(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_IQ4_XS
        Dequant.dequantize_iq4_xs(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q2_K
        Dequant.dequantize_q2_k(@view(file.tensor_data[start:end]), num_elements)
    elseif info.type == GGUF.GGML_TYPE_Q3_K
        Dequant.dequantize_q3_k(@view(file.tensor_data[start:end]), num_elements)
  elseif info.type == GGUF.GGML_TYPE_Q6_K
 num_blocks = num_elements ÷ 256
 data_size = num_blocks * QuantsCPU.Q6_K_BLOCK_SIZE
 Dequant.dequantize_q6_k(collect(@view file.tensor_data[start:start+data_size-1]), num_elements)
    else
        @warn "Unhandled type, zeroing" type = info.type
        zeros(Float32, num_elements)
    end

 if length(dims) > 2
 return reshape(data, dims)
 else
 # GGUF stores data as [inner][outer] where inner varies fastest
 # For weight matrices, we typically want (outer, inner) for mat-vec mul
 # But embeddings and norms should stay as (inner, outer)
 # Check if this looks like a weight matrix (usually has names like "weight")
 # Embedding matrices and special tensors should not be transposed
 if length(dims) == 2
 # For now, try without transpose and see what breaks
 return reshape(data, inner, outer)
 else
 return reshape(data, dims)
 end
 end
end

function extract_tensor_cpu(file::GGUF.GGUFFile, name::String)
 if !haskey(file.tensors, name)
 @warn "Tensor not found, using zero surrogate" name = name
 return zeros(Float32, 1, 1)
 end
 info = file.tensors[name]
 return extract_tensor_cpu(file, info)
end

"""
 load_model_cpu(path::String; keep_quantized=false) -> (QwenModelCPU, tokenizer)

Load a GGUF or Safetensors model for CPU inference.

Detects model format automatically (GGUF file or Safetensors directory)
and loads the appropriate backend.

# Arguments
- `path::String`: Path to GGUF file, safetensors file, or directory containing model
- `keep_quantized`: If true, preserve quantized weights for memory efficiency. If nothing (default), auto-detect based on C kernel availability.

# Returns
- `model::QwenModelCPU`: The loaded CPU model
- `tokenizer`: The loaded BPETokenizer

# Example
```julia
# Load GGUF model
model, tok = load_model_cpu("model.gguf")

# Load from directory (auto-detects safetensors)
model, tok = load_model_cpu("path/to/model_dir")

# Keep quantized weights for Q4_K models
model, tok = load_model_cpu("model.gguf"; keep_quantized=true)
```

# Supported Formats
- GGUF (.gguf files) - llama.cpp format
- Safetensors - HuggingFace format
"""
function load_model_cpu(path::String; keep_quantized::Union{Bool,Nothing}=nothing, use_bf16_weights::Bool=false)
 # Detect format
 fmt = detect_model_format(path)
 
# Pre-load Qwen3 module for architecture detection
 
if fmt == :safetensors
# Check architecture for safetensors models before loading
try
config_json = joinpath(path, "config.json")
if isfile(config_json)
config_text = read(config_json, String)
if occursin("\"qwen3\"", config_text) || occursin("\"Qwen3ForCausalLM\"", config_text)
@info "Detected Qwen3 architecture - routing to Qwen3 loader"
return Qwen3.load_qwen3_safetensors(path; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
elseif occursin("\"jamba\"", config_text) || occursin("\"JambaForCausalLM\"", config_text)
@info "Detected Jamba architecture - routing to Jamba loader"
return Jamba.load_jamba_safetensors(path; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
end
end
catch e
@warn "Error checking safetensors architecture" exception=(e, catch_backtrace())
end
# Fall through to existing safetensors loader
return load_safetensors_model(path)
 elseif fmt == :gguf
 # If path is a directory, find the GGUF file
 original_path = path
 if isdir(path)
 files = readdir(path)
 gguf_files = filter(f -> endswith(lowercase(f), ".gguf"), files)
 if isempty(gguf_files)
 error("No GGUF files found in directory: $path")
 end
# First check if this is a Qwen3 GGUF by loading metadata only
# Load GGUF header to check architecture
file = GGUF.read_gguf(path)
arch = get(file.metadata, "general.architecture", "qwen")
if arch == "qwen3"
@info "Detected Qwen3 GGUF architecture - routing to Qwen3 loader"
return Qwen3.load_qwen3_cpu(path; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
elseif arch == "jamba"
@info "Detected Jamba GGUF architecture - routing to Jamba loader"
return Jamba.load_jamba_cpu(path; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
end
# Not Qwen3/Jamba, continue with normal loading - path is still directory
 # Prefer F32 or F16 over quantized versions
 preferred = ["F32.gguf", "F16.gguf", "BF16.gguf"]
 selected_file = nothing
 for pref in preferred
 matches = filter(f -> occursin(pref, f), gguf_files)
 if !isempty(matches)
 selected_file = matches[1]
 break
 end
 end
 if selected_file === nothing
 # Use first GGUF file
 selected_file = gguf_files[1]
 end
 path = joinpath(original_path, selected_file)
 end
 else
 error("Unknown model format for path: $path")
 end
 
 println("Loading GGUF: $path")
 
 # Load GGUF file
 file = GGUF.read_gguf(path)
    
    # Get config
    config = get_config(file)
    
    println("Config: hidden=$(config.hidden_size), layers=$(config.num_hidden_layers), heads=$(config.num_attention_heads)")
    
    # Load embedding
    embed = extract_tensor_cpu(file, "token_embd.weight")
    embed = Float32.(embed)
    
    # Get actual vocab size from embedding (may differ from metadata due to special tokens)
    actual_vocab_size = size(embed, 2)
    
    println("Embedding: $(size(embed))")
    
    # Update config with actual vocab size
    config = ModelCPU.QwenConfigCPU(
        architecture = config.architecture,
        vocab_size = actual_vocab_size,
        hidden_size = config.hidden_size,
        intermediate_size = config.intermediate_size,
        num_hidden_layers = config.num_hidden_layers,
        num_attention_heads = config.num_attention_heads,
        num_key_value_heads = config.num_key_value_heads,
        head_dim = config.head_dim,
        rms_norm_eps = config.rms_norm_eps,
        rope_theta = config.rope_theta,
        max_position_embeddings = config.max_position_embeddings,
        full_attention_interval = config.full_attention_interval,
        ssm_inner_size = config.ssm_inner_size,
        ssm_state_size = config.ssm_state_size,
        ssm_group_count = config.ssm_group_count,
        ssm_time_step_rank = config.ssm_time_step_rank,
        ssm_conv_kernel = config.ssm_conv_kernel,
        partial_rotary_factor = config.partial_rotary_factor
    )
    
 if use_bf16_weights && ArrowLake.has_arrow_lake_features()
 @info "BF16 weight conversion enabled for Arrow Lake - 50% memory reduction on weights"
 end
 
 # Auto-detect keep_quantized: default to F32 BLAS unless memory-constrained
 # C kernels are slower than BLAS for most matrix sizes on this model
 if keep_quantized === nothing
 keep_quantized = false
 if QuantizedKernels.quant_kernels_available()
 println("Dequantizing to F32 (BLAS is faster for this model size — use keep_quantized=true for memory savings)")
 else
 println("Dequantizing to F32 (C kernels not available)")
 end
 end
 
 # Load layers
 layers = ModelCPU.DecoderLayerCPU[]
 
 for i in 0:(config.num_hidden_layers - 1)
 layer = load_layer(file, i, config; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
 push!(layers, layer)
 println(" Layer $i: $(layer.is_ssm ? "SSM" : "Attention")")
 end
 
 ModelCPU.init_layer_counters!(length(layers))
 
 # Load final norm
    final_norm_w = Float32.(extract_tensor_cpu(file, "output_norm.weight"))
    final_norm = ModelCPU.RMSNormCPU(final_norm_w, config.rms_norm_eps)
    
 # LM head (check if tied or separate)
 # For BF16: store in ROW-MAJOR (transposed) so C kernel has contiguous rows
 if haskey(file.tensors, "output.weight")
 lm_head_raw = Float32.(extract_tensor_cpu(file, "output.weight"))
 if use_bf16_weights && ArrowLake.has_arrow_lake_features()
 lm_head = Matrix{BFloat16}(permutedims(BFloat16.(Matrix{Float32}(lm_head_raw))))
 else
 lm_head = Matrix{Float32}(lm_head_raw)
 end
 else
 # Tied weights - still convert to BF16 if requested (separate copy from embed)
 if use_bf16_weights && ArrowLake.has_arrow_lake_features()
 lm_head = Matrix{BFloat16}(permutedims(BFloat16.(Matrix{Float32}(embed'))))
 else
 lm_head = Matrix{Float32}(embed')
 end
 end
 
 # Create RoPE with partial rotary
 rotary_dim = round(Int, config.head_dim * config.partial_rotary_factor)
 rope = ModelCPU.RotaryEmbeddingCPU(config.head_dim, config.rope_theta, config.max_position_embeddings; rotary_dim=rotary_dim)
 
 # Pre-allocate buffers for final_norm and lm_head to avoid per-token allocations
 embed_buf = Vector{Float32}(undef, config.hidden_size)
 final_norm_buf = Vector{Float32}(undef, config.hidden_size)
 lm_head_buf = Vector{Float32}(undef, config.vocab_size)
 
 # Load tokenizer
 tok = Tokenizer.load_tokenizer(file.metadata)
 
 # GGUF files don't include MTP weights - they're skipped during conversion
 # MTP is only available in safetensors format
 mtp_head = nothing
 
 return ModelCPU.QwenModelCPU(config, embed, lm_head, layers, final_norm, rope, embed_buf, final_norm_buf, lm_head_buf, mtp_head), tok
end

function get_config(file::GGUF.GGUFFile)
    arch = get(file.metadata, "general.architecture", "qwen")
    
    config = ModelCPU.QwenConfigCPU(
        architecture = Symbol(arch),
        vocab_size = get(file.metadata, "qwen3.vocab_size", get(file.metadata, "vocab_size", 151936)),
        hidden_size = get(file.metadata, "qwen3.embedding_length", get(file.metadata, "embedding_length", 1024)),
        intermediate_size = get(file.metadata, "qwen3.feed_forward_length", get(file.metadata, "feed_forward_length", 3584)),
        num_hidden_layers = get(file.metadata, "qwen3.block_count", get(file.metadata, "block_count", 24)),
        num_attention_heads = get(file.metadata, "qwen3.attention.head_count", get(file.metadata, "attention.head_count", 8)),
        num_key_value_heads = get(file.metadata, "qwen3.attention.head_count_kv", get(file.metadata, "attention.head_count_kv", 2)),
        head_dim = get(file.metadata, "qwen3.attention.key_length", get(file.metadata, "attention.key_length", 256)),
        rms_norm_eps = Float32(get(file.metadata, "qwen3.attention.layer_norm_rms_epsilon", get(file.metadata, "attention.layer_norm_rms_epsilon", 1e-6))),
        rope_theta = Float32(get(file.metadata, "qwen3.rope.freq_base", get(file.metadata, "rope.freq_base", 10000000.0))),
        max_position_embeddings = get(file.metadata, "qwen3.context_length", get(file.metadata, "context_length", 4096)),
        partial_rotary_factor = Float32(get(file.metadata, "qwen3.rope.partial_rotary_factor", get(file.metadata, "rope.partial_rotary_factor", 0.25))),
        full_attention_interval = get(file.metadata, "qwen3.full_attention_interval", get(file.metadata, "full_attention_interval", 4)),
        ssm_inner_size = get(file.metadata, "qwen3.ssm.inner_size", get(file.metadata, "ssm.inner_size", 2048)),
        ssm_state_size = get(file.metadata, "qwen3.ssm.state_size", get(file.metadata, "ssm.state_size", 128)),
        ssm_group_count = get(file.metadata, "qwen3.ssm.group_count", get(file.metadata, "ssm.group_count", 16)),
        ssm_time_step_rank = get(file.metadata, "qwen3.ssm.time_step_rank", get(file.metadata, "ssm.time_step_rank", 16)),
        ssm_conv_kernel = get(file.metadata, "qwen3.ssm.conv_kernel", get(file.metadata, "ssm.conv_kernel", 4)),
    )
    
    return config
end

function load_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU; keep_quantized::Bool=false, use_bf16_weights::Bool=false)
 prefix = "blk.$(layer_idx)"
 
 # Load norms (always Float32 for numerical stability)
 in_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_norm.weight"))
 in_norm = ModelCPU.RMSNormCPU(in_norm_w, config.rms_norm_eps)
 
 post_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).post_attention_norm.weight"))
 post_norm = ModelCPU.RMSNormCPU(post_norm_w, config.rms_norm_eps)
 
 # Check if SSM layer - it has ssm_a tensor
 is_ssm = haskey(file.tensors, "$(prefix).ssm_a")
 
 if is_ssm
 op = load_ssm_layer(file, layer_idx, config; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
 else
 op = load_attention_layer(file, layer_idx, config; use_bf16_weights=use_bf16_weights)
 end
 
 # Load MLP
 mlp = load_mlp(file, layer_idx, config; keep_quantized=keep_quantized, use_bf16_weights=use_bf16_weights)
 
 # Pre-allocate norm buffers
 hidden_size = config.hidden_size
 norm_buf1 = zeros(Float32, hidden_size)
 norm_buf2 = zeros(Float32, hidden_size)
 
 return ModelCPU.DecoderLayerCPU(in_norm, op, post_norm, mlp, is_ssm, norm_buf1, norm_buf2)
 end

function load_ssm_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU; keep_quantized::Bool=false, use_bf16_weights::Bool=false)
 prefix = "blk.$(layer_idx)"
 
 # Get tensor info for weight type checking
 in_proj_info = file.tensors["$(prefix).attn_qkv.weight"]
 gate_info = file.tensors["$(prefix).attn_gate.weight"]
 ssm_out_info = file.tensors["$(prefix).ssm_out.weight"]
 
 # Load weights - note: GGUF stores as (out_features, in_features), we need to transpose
 # For quantized weights, we keep them quantized and handle transpose in multiplication
 
 if keep_quantized && in_proj_info.type in (GGUF.GGML_TYPE_Q4_K, GGUF.GGML_TYPE_Q5_K, 
 GGUF.GGML_TYPE_Q6_K, GGUF.GGML_TYPE_Q8_0)
 # Load quantized weights - no transpose needed, handled in multiplication
 in_proj = extract_tensor_cpu(file, in_proj_info; keep_quantized=true)
 gate_proj = extract_tensor_cpu(file, gate_info; keep_quantized=true)
 ssm_out = extract_tensor_cpu(file, ssm_out_info; keep_quantized=true)
 else
 # Dequantize and transpose (always F32 - BLAS is faster for compute-bound ops)
 in_proj = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).attn_qkv.weight"))')
 gate_proj = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).attn_gate.weight"))')
 ssm_out = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ssm_out.weight"))')
 end
    
# Conv1d - GGUF stores as (K, C) = (4, 6144) row-major
# Python uses (C, K) = (6144, 4) for access as [:, k]
# We need transpose to match: Julia ssm_conv1d' should give (C, K)
ssm_conv1d_raw = extract_tensor_cpu(file, "$(prefix).ssm_conv1d.weight")
# After extract_tensor_cpu, raw is (K, C) = (4, 6144)
# We need to transpose to (C, K) = (6144, 4) to match Python
ssm_conv1d = Matrix{Float32}(ssm_conv1d_raw')  # Transpose to (6144, 4) = (C, K)
 
# Alpha/beta weights
# GGUF stores as (num_v_heads, hidden_size) in row-major
# We need (num_v_heads, hidden_size) for the forward pass which does weight' * x
# So we transpose at load: (hidden, heads) -> (heads, hidden) so that (heads, hidden)' = (hidden, heads)
ssm_alpha_weight = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ssm_alpha.weight"))')
# After transpose we have (num_v_heads, hidden_size) - no transpose needed in forward pass
ssm_beta_weight = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ssm_beta.weight"))')
    
    # SSM parameters - ssm_a is a tensor, not a bias
    ssm_a = Float32.(vec(extract_tensor_cpu(file, "$(prefix).ssm_a")))
    ssm_dt_bias = Float32.(vec(extract_tensor_cpu(file, "$(prefix).ssm_dt.bias")))
    
    # SSM norm
    ssm_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).ssm_norm.weight"))
    ssm_norm = ModelCPU.RMSNormCPU(ssm_norm_w, config.rms_norm_eps)
 
 # Dimensions - need to get from model
 # For Qwen3.5-0.8B: d_inner = 2048, num_v_heads = 16, num_k_heads = 16
 # head_k_dim = 128, head_v_dim = 128
 num_v_heads = length(ssm_a)
 num_k_heads = num_v_heads # same for Qwen3.5
 
 # Get dimensions - handle both Float32 and quantized weights
 d_inner = weight_outer_dim(gate_proj)
 head_v_dim = d_inner ÷ num_v_heads
 head_k_dim = size(ssm_alpha_weight, 1) ÷ num_k_heads # Actually need to check this
 
# conv_channels = d_inner + 2 * num_k_heads * head_k_dim
    conv_channels = weight_outer_dim(in_proj) # 6144 (C)
    conv_kernel = size(ssm_conv1d, 2) # 4 (K)
    
 # Recompute head_k_dim from conv_channels
 # conv_channels = d_inner + 2 * num_k_heads * head_k_dim
 # head_k_dim = (conv_channels - d_inner) / (2 * num_k_heads)
 head_k_dim = (conv_channels - d_inner) ÷ (2 * num_k_heads)
 
 # State buffers
 conv_state = zeros(Float32, conv_channels, conv_kernel)
 h = zeros(Float32, head_v_dim, head_k_dim, num_v_heads) # Per-group states
 
 # Pre-allocated work buffers to avoid GC
 x_conv_buf = Vector{Float32}(undef, conv_channels)
 y_all_buf = Vector{Float32}(undef, d_inner)
 alpha_proj_buf = Vector{Float32}(undef, num_v_heads)
 beta_proj_buf = Vector{Float32}(undef, num_v_heads)
 # Pre-allocated mat-vec output buffers
 qkv_buf = Vector{Float32}(undef, conv_channels)
 z_buf = Vector{Float32}(undef, d_inner)
 out_buf = Vector{Float32}(undef, config.hidden_size)
 # Per-head loop buffers
 sk_buf = Vector{Float32}(undef, head_v_dim)
 d_buf = Vector{Float32}(undef, head_v_dim)
 y_h_buf = Vector{Float32}(undef, head_v_dim)
 q_norm_buf = Vector{Float32}(undef, head_k_dim)
 k_norm_buf = Vector{Float32}(undef, head_k_dim)
 
 return ModelCPU.GatedDeltaNetCPU(
 layer_idx + 1,
 in_proj, gate_proj, ssm_out, ssm_conv1d,
 ssm_alpha_weight, ssm_beta_weight,
 ssm_a, ssm_dt_bias, ssm_norm,
 num_v_heads, num_k_heads, head_k_dim, head_v_dim, d_inner,
 conv_channels, conv_kernel,
 conv_state, h,
 x_conv_buf, y_all_buf, alpha_proj_buf, beta_proj_buf,
 qkv_buf, z_buf, out_buf,
 sk_buf, d_buf, y_h_buf,
 q_norm_buf, k_norm_buf
 )
end

function load_attention_layer(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU; use_bf16_weights::Bool=false)
 prefix = "blk.$(layer_idx)"
 
 # Load weights - transpose from GGUF format (always F32)
 wq = collect(Float32.(extract_tensor_cpu(file, "$(prefix).attn_q.weight"))')
 wk = collect(Float32.(extract_tensor_cpu(file, "$(prefix).attn_k.weight"))')
 wv = collect(Float32.(extract_tensor_cpu(file, "$(prefix).attn_v.weight"))')
 wo = collect(Float32.(extract_tensor_cpu(file, "$(prefix).attn_output.weight"))')
    
    # Q/K norms
    q_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_q_norm.weight"))
    q_norm = ModelCPU.RMSNormCPU(q_norm_w, config.rms_norm_eps)
    
    k_norm_w = Float32.(extract_tensor_cpu(file, "$(prefix).attn_k_norm.weight"))
    k_norm = ModelCPU.RMSNormCPU(k_norm_w, config.rms_norm_eps)
    
    # Get head dimensions from weights
    # wq projects to n_heads * head_dim * 2 (query + gate)
    # So n_heads = size(wq, 1) ÷ head_dim ÷ 2
    n_heads = size(wq, 1) ÷ config.head_dim ÷ 2
    n_kv = size(wk, 1) ÷ config.head_dim
    
    scale = 1.0f0 / sqrt(Float32(config.head_dim))
 
 # Pre-allocated work buffers
 qkv_size = n_heads * config.head_dim * 2  # wq output size
 kv_size = n_kv * config.head_dim           # wk/wv output size
 q_size = n_heads * config.head_dim         # query after split
 
 qkv_buf = Vector{Float32}(undef, qkv_size)
 k_buf = Vector{Float32}(undef, kv_size)
 v_buf = Vector{Float32}(undef, kv_size)
 query_states_buf = Vector{Float32}(undef, q_size)
 gate_buf = Vector{Float32}(undef, q_size)
 output_buf = Vector{Float32}(undef, q_size)
 wo_output_buf = Vector{Float32}(undef, config.hidden_size)
 # Max seq_len for scores buffer
 max_seq = config.max_position_embeddings
 scores_buf = Vector{Float32}(undef, max_seq)
 
 # Flash attention output buffer (head_dim per head)
 fa_output_buf = Vector{Float32}(undef, config.head_dim)
 
 # Default: flash attention disabled (benchmark first)
 use_fa = get(ENV, "INFERNO_USE_FLASH_ATTENTION", "true") != "false"
 
 return ModelCPU.FullAttentionCPU(
 layer_idx + 1,
 wq, wk, wv, wo,
 q_norm, k_norm,
 n_heads, n_kv, config.head_dim, scale,
 qkv_buf, k_buf, v_buf,
 query_states_buf, gate_buf, output_buf, scores_buf, wo_output_buf,
 fa_output_buf, use_fa
 )
end

function load_mlp(file::GGUF.GGUFFile, layer_idx::Int, config::ModelCPU.QwenConfigCPU; keep_quantized::Bool=false, use_bf16_weights::Bool=false)
 prefix = "blk.$(layer_idx)"
 
 # Load weights - transpose from GGUF format
 # For quantized weights, we store them directly and handle transpose in multiplication
 gate_info = file.tensors["$(prefix).ffn_gate.weight"]
 up_info = file.tensors["$(prefix).ffn_up.weight"]
 down_info = file.tensors["$(prefix).ffn_down.weight"]
 
 # Determine intermediate size from weight dimensions
 # gate_weight will be (intermediate, hidden) after transpose
 intermediate_size = config.intermediate_size
 
 # Pre-allocated work buffers
 gate_buf = Vector{Float32}(undef, intermediate_size)
 up_buf = Vector{Float32}(undef, intermediate_size)
 hidden_buf = Vector{Float32}(undef, intermediate_size)
 output_buf = Vector{Float32}(undef, config.hidden_size)
 
 if keep_quantized && gate_info.type in (GGUF.GGML_TYPE_Q4_K, GGUF.GGML_TYPE_Q5_K, 
 GGUF.GGML_TYPE_Q6_K, GGUF.GGML_TYPE_Q8_0)
 # Load quantized weights - no transpose, we'll handle it in multiplication
 gate_weight = extract_tensor_cpu(file, gate_info; keep_quantized=true)
 up_weight = extract_tensor_cpu(file, up_info; keep_quantized=true)
 down_weight = extract_tensor_cpu(file, down_info; keep_quantized=true)
 return ModelCPU.MLPCPU(gate_weight, up_weight, down_weight, gate_buf, up_buf, hidden_buf, output_buf)
 else
 # Dequantize to Float32 and transpose (always F32 - BLAS is faster for compute-bound ops)
 gate_weight = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ffn_gate.weight"))')
 up_weight = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ffn_up.weight"))')
 down_weight = Matrix(Float32.(extract_tensor_cpu(file, "$(prefix).ffn_down.weight"))')
 return ModelCPU.MLPCPU(gate_weight, up_weight, down_weight, gate_buf, up_buf, hidden_buf, output_buf)
 end
end

end # module
