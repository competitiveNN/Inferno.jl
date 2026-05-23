module Inferno

using Printf

# GPU backend (optional)
try
    using oneAPI
catch
end

include("QuantsData.jl")
include("Dequant.jl")
include("AMXBF16.jl")
include("ArrowLake.jl")
include("QuantsCPU.jl")
include("QuantMV.jl")
include("QuantizedKernels.jl")
include("GGUF.jl")
include("Model.jl")
include("BF16Support.jl")
include("CommonOps.jl")
include("ModelCPU.jl")
include("Tokenizer.jl")
include("Loader.jl")
include("Safetensors.jl")
include("Qwen3.jl")
include("Qwen35.jl")
include("Jamba.jl")
include("LoaderCPU.jl")
include("Gemma4.jl")
include("Gemma4Loader.jl")
include("GPUCommon.jl")
include("FusedKernels.jl")
include("Qwen35GPU.jl")
include("Gemma4GPUKernels.jl")
include("Gemma4GPU.jl")
include("Gemma4GPULoader.jl")
include("Engine.jl")
include("Server.jl")
include("Generate.jl")

using .QuantsData
using .Dequant
using .Gemma4
using .Gemma4Loader
using .Gemma4GPU
using .Gemma4GPULoader
using .GPUCommon
using .FusedKernels
using .Qwen35GPU
using .Engine
using .AMXBF16
using .ArrowLake
using .QuantsCPU
using .QuantMV
using .GGUF
using .Model
using .BF16Support: set_inference_precision!, get_inference_precision, should_use_bf16
using .ModelCPU
using .Jamba
using .Tokenizer
using .Loader
using .LoaderCPU
using .Safetensors
using .Qwen3
using .Server
using .Generate

export load_model, load_model_cpu, start_server, non_nothing_fields, stream_to_stdout, stream_to_stdout_cpu
export load_qwen3_cpu, generate_stream_cpu
export LoaderCPU, ModelCPU, generate_stream_cpu, generate_cpu, softmax_sample, BF16Support
export generate_text
export chat!, start_chat, Message, build_prompt
export load_safetensors_model, Safetensors, detect_model_format
export set_inference_precision!, get_inference_precision, should_use_bf16

"""
 non_nothing_fields(obj) -> NamedTuple

Return a NamedTuple containing all fields of `obj` that are not `nothing`.
Useful for debugging and inspecting model structures.

# Example
```julia
model = load_model_cpu("model.gguf")
fields = non_nothing_fields(model)
println("Loaded model has ", length(fields), " non-nothing fields")
```
"""
function non_nothing_fields(obj)
    fns = fieldnames(typeof(obj))
    names = Symbol[]
    vals = Any[]
    for f in fns
        val = getfield(obj, f)
        if !isnothing(val)
            push!(names, f)
            push!(vals, val)
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(vals))
end

# Probe a device for the ZE_RESULT_ERROR_UNKNOWN driver corruption bug.
# Per JuliaGPU/oneAPI.jl#458: Intel Arc driver permanently corrupts the L0 command
# queue across process boundaries after mixed HostBuffer/DeviceBuffer kernel ops.
# A small probe (4 bytes) passes even on corrupted devices; medium probes (256KB) detect
# devices that fail quickly. Large uploads (>1MB) may hang indefinitely on some states.
# Returns :ok, :fast_fail, or :unknown
function probe_device(dev)::Symbol
    try
        oneAPI.device!(dev)
        # Small probe to detect immediate failures
        arr = oneArray(Int32[1])
        oneAPI.unsafe_free!(arr)
        # Medium probe to detect partial failures (256KB)
        arr2 = oneArray(zeros(Float16, 64 * 1024))
        oneAPI.unsafe_free!(arr2)
        return :ok
    catch
        return :fast_fail
    end
end

function select_device!(devs, requested_device)
    if !isnothing(requested_device)
        # Use specific requested device
        target_idx = clamp(requested_device, 1, length(devs))
        oneAPI.device!(devs[target_idx])
        @info "Using requested GPU" target_idx device=string(oneAPI.device())
        return target_idx
    end

    # Auto-select: start from second GPU, probe devices to find working one
    # Default to GPU 2 (index 2) if available, else try others
    start_idx = min(2, length(devs))
    for offset in 0:(length(devs)-1)
        i = mod1(start_idx + offset, length(devs))
        dev = devs[i]
        status = probe_device(dev)
        if status == :ok
            oneAPI.device!(dev)
            @info "GPU probe successful" index=i device=string(oneAPI.device())
            return i
        else
            @warn "GPU probe failed" index=i status=status
        end
    end

    # Fallback to first device if all fail
    target_idx = 1
    oneAPI.device!(devs[target_idx])
    @warn "All GPUs failed probe, using fallback" target_idx device=string(oneAPI.device())
    return target_idx
end

"""
 load_model(path; device=nothing, mmproj=nothing, backend=:auto) -> (model, tokenizer)

Load a GGUF or Safetensors model file and return the constructed model + tokenizer.

# Arguments
- `path::String`: Path to the model file (.gguf) or directory (.safetensors)
- `device::Union{Int,Nothing}`: GPU device index to use (nothing for auto-detect)
- `mmproj::Union{String,Nothing}`: Path to multimodal projection weights (auto-discovered if not provided)
- `backend::Symbol`: Backend to use (`:auto`, `:cpu`, or `:gpu`)

# Returns
- `model`: The loaded QwenModel (GPU) or QwenModelCPU (CPU)
- `tokenizer`: The loaded BPETokenizer

# Example
```julia
model, tok = load_model("path/to/model.gguf")
model, tok = load_model("path/to/model_dir"; backend=:cpu)  # Force CPU
```

# Notes
- Safetensors format only supports CPU backend currently
- GPU backend requires oneAPI-compatible Intel GPU
"""
function find_related_file(model_path::String, pattern::String)
    model_dir = dirname(model_path)
    if isempty(model_dir)
        model_dir = "."
    end

    # Try exact match first
    exact_path = joinpath(model_dir, pattern)
    if isfile(exact_path)
        return exact_path
    end

    # Try case-insensitive substring match
    try
        files = readdir(model_dir)
        p_lower = lowercase(pattern)
        for f in files
            f_lower = lowercase(f)
            if occursin(p_lower, f_lower)
                return joinpath(model_dir, f)
            end
        end
    catch
    end

    return nothing
end

function load_model(path::String; device::Union{Int, Nothing}=nothing,
 mmproj::Union{String, Nothing}=nothing,
 backend::Symbol=:auto)
 
 # Detect model format
 fmt = LoaderCPU.detect_model_format(path)
 
 # Handle backend selection
 if backend == :cpu
 if fmt == :safetensors
 return Safetensors.load_safetensors_model(path)
 else
 return LoaderCPU.load_model_cpu(path)
 end
 end
 
 # For GPU backend, only GGUF is supported currently
 if fmt == :safetensors
 @warn "Safetensors format only supports CPU backend. Use backend=:cpu for safetensors models."
 return Safetensors.load_safetensors_model(path)
 end

    model = nothing
    tok = nothing

    devs = collect(oneAPI.devices())
    selected_device_idx = 1
    if isempty(devs)
        @warn "No oneAPI devices found."
    else
        selected_device_idx = select_device!(devs, device)
    end

    @info "Loading GGUF file" path

    # Automate discovery if paths are not provided
    if mmproj === nothing
        mmproj = find_related_file(path, "mmproj")
        if mmproj !== nothing
            @info "Auto-discovered mmproj" mmproj
        end
    end

    file = nothing
    try
        file = GGUF.read_gguf(path)
        @info "GGUF file loaded" metadata_keys=length(file.metadata) tensors=length(file.tensors)

        arch_str = get(file.metadata, "general.architecture", "qwen2")
        arch = Symbol(arch_str)

        config = Model.QwenConfig(
            architecture=arch,
            vocab_size=Int(get(file.metadata, "$(arch_str).vocab_size",
                length(get(file.metadata, "tokenizer.ggml.tokens", [])))),
            hidden_size=Int(get(file.metadata, "$(arch_str).embedding_length", 1024)),
            intermediate_size=Int(get(file.metadata, "$(arch_str).feed_forward_length", 3584)),
            num_hidden_layers=Int(get(file.metadata, "$(arch_str).block_count", 24)),
            num_attention_heads=Int(get(file.metadata, "$(arch_str).attention.head_count", 8)),
            num_key_value_heads=Int(get(file.metadata, "$(arch_str).attention.head_count_kv", 2)),
            head_dim=Int(get(file.metadata, "$(arch_str).attention.key_length", 256)),
 rms_norm_eps=Float32(get(file.metadata, "$(arch_str).attention.layer_norm_rms_epsilon", 1.0e-6)),
 rope_theta=Float32(get(file.metadata, "$(arch_str).rope.freq_base", 10000000.0)),
            max_position_embeddings=min(4096, Int(get(file.metadata, "$(arch_str).context_length", 32768))),
            full_attention_interval=Int(get(file.metadata, "$(arch_str).full_attention_interval", 4)),
            ssm_inner_size=Int(get(file.metadata, "$(arch_str).ssm.inner_size", 2048)),
            ssm_state_size=Int(get(file.metadata, "$(arch_str).ssm.state_size", 128)),
            ssm_group_count=Int(get(file.metadata, "$(arch_str).ssm.group_count", 16)),
            ssm_time_step_rank=Int(get(file.metadata, "$(arch_str).ssm.time_step_rank", 16)),
            ssm_conv_kernel=Int(get(file.metadata, "$(arch_str).ssm.conv_kernel", 4)),

            # MoE
            num_experts=Int(get(file.metadata, "$(arch_str).expert_count", 0)),
            num_experts_per_tok=Int(get(file.metadata, "$(arch_str).expert_used_count", 0)),

            # MLA
            q_lora_rank=Int(get(file.metadata, "$(arch_str).attention.q_lora_rank", 0)),
            kv_lora_rank=Int(get(file.metadata, "$(arch_str).attention.kv_lora_rank", 0)),
            qk_rope_head_dim=Int(get(file.metadata, "$(arch_str).attention.qk_rope_head_dim", 0)),
            v_head_dim=Int(get(file.metadata, "$(arch_str).attention.v_head_dim", 0)),
        )
        @info "Model config created" hidden=config.hidden_size layers=config.num_hidden_layers heads=config.num_attention_heads

        @info "Initializing GPU tables"
        Model.init_gpu_tables(QuantsData.IQ2XXS_GRID, QuantsData.KSIGNS_IQ2XS, QuantsData.KMASK_IQ2XS)


        mmproj_file = nothing
        if mmproj !== nothing
            @info "Loading mmproj" mmproj
            mmproj_file = GGUF.read_gguf(mmproj)
        end

        model = Loader.load_weights(file, config; mmproj=mmproj_file)
        @info "Model weights loaded"

        @info "Loading tokenizer"
        tok = Tokenizer.load_tokenizer(file.metadata)
        @info "Tokenizer loaded" vocab_size=length(tok.id_to_token) bos=tok.bos_id eos=tok.eos_id

        return model, tok
    catch e
        @error "ERROR loading model" exception=(e, catch_backtrace())
        # Cleanup on error
        try
            if model !== nothing
                Model.free_model_gpu!(model)
            end
            Model.free_gpu_tables!()
        catch
        end
        try
            GC.gc(true)
        catch
        end
        try
            oneAPI.synchronize()
        catch
        end
        rethrow(e)
    end
end

"""
 main(model_path; port=8080, device=nothing, auth_token=nothing, backend=:auto)

Load model and start the HTTP OpenAI-compatible API server.

# Arguments
- `model_path::String`: Path to the model file or directory
- `port::Int=8080`: Port to listen on
- `device::Union{Int,Nothing}=nothing`: GPU device index (nil for auto-detect)
- `auth_token::Union{String,Nothing}=nothing`: Optional bearer token for authentication
- `backend::Symbol=:auto`: Backend selection (`:auto`, `:cpu`, `:gpu`)

# Example
```julia
main("model.gguf"; port=8080)
main("model.gguf"; port=8080, auth_token="secret123")
```
"""
function main(model_path::String; port::Int=8080, device::Union{Int, Nothing}=nothing, auth_token::Union{String, Nothing}=nothing, backend::Symbol=:auto)
    model, tok = load_model(model_path; device=device, backend=backend)
    Server.start_server(port; model=model, tokenizer=tok, auth_token=auth_token)
end

"""
 to_float16(x) -> Float16

Convert any real number to Float16. Handles Float32, Float64, and Integer types.

This is an internal helper function used for converting sampling parameters
to Float16 for GPU backend compatibility.
"""
to_float16(x::Float16) = x
to_float16(x::Float32) = Float16(x)
to_float16(x::Float64) = Float16(x)
to_float16(x::Integer) = Float16(x)
to_float16(x::AbstractFloat) = Float16(x)

"""
 stream_to_stdout(model, tok, prompt; kwargs...)

Generate text from a prompt and stream tokens to stdout as they are generated.
Returns the complete generated text as a string.

This is a convenience wrapper that automatically converts Float32/Float64/Int 
parameters to Float16 for compatibility with the GPU inference engine.

# Arguments
- `model`: The loaded QwenModel
- `tok`: The BPETokenizer
- `prompt`: The input prompt string
- `max_tokens`: Maximum number of tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 0.7) - accepts Float16/Float32/Float64
- `top_p`: Nucleus sampling probability (default: 0.8) - accepts Float16/Float32/Float64
- `top_k`: Top-k sampling parameter (default: 20)
- `presence_penalty`: Penalty for repeated tokens (default: 0.0) - accepts Float16/Float32/Float64
- `repetition_penalty`: Multiplicative repetition penalty (default: 1.0) - accepts Float16/Float32/Float64
- `io`: Output IO stream (default: stdout)
- `use_bf16`: Use BF16 precision for weights on CPU (default: false). Requires BF16 model.

# Presence & Repetition Penalties
- `presence_penalty` is an additive penalty applied proportional to how many times a token has appeared.
- `repetition_penalty` is a multiplicative penalty applied to logits of previously seen tokens (common default: 1.0; use 1.1-1.2 to discourage repetition).

# Examples
```julia
# Using Float64 (automatically converted)
result = stream_to_stdout(model, tok, "Hello"; temperature=0.7, repetition_penalty=1.1)

# Using Float32
result = stream_to_stdout(model, tok, "Hello"; temperature=0.7f0, repetition_penalty=1.1f0)

# Using Float16 directly
result = stream_to_stdout(model, tok, "Hello"; temperature=Float16(0.7), repetition_penalty=Float16(1.1))

# Using BF16 precision (CPU only, requires BF16 model)
result = stream_to_stdout(model, tok, "Hello"; use_bf16=true, temperature=0.7f0)
```
"""
function stream_to_stdout(model, tok, prompt::AbstractString;
 backend::Symbol=:auto,
 max_tokens::Int=100,
 temperature=0.7,
 top_p=0.8,
 top_k::Int=20,
 presence_penalty=0.0,
 repetition_penalty=1.0,
 min_p=0.0,
 stop_token::Union{Int,Nothing}=nothing,
 show_tps::Bool=false,
 use_bf16::Bool=false,
 io::IO=stdout,
 interrupt_check::Function=() -> false)
    
    # Auto-select backend when not specified, prefer model type
    chosen_backend = backend
    if backend == :auto
        if model isa Model.QwenModel
            chosen_backend = :gpu
        elseif model isa ModelCPU.QwenModelCPU
            chosen_backend = :cpu
        else
            error("stream_to_stdout: unable to infer backend from model type $(typeof(model)). Provide backend=:gpu or :cpu explicitly.")
        end
    end

    if chosen_backend == :gpu
        if !(tok isa Tokenizer.BPETokenizer)
            error("GPU backend requires a Tokenizer.BPETokenizer. Provided tokenizer is $(typeof(tok)).")
        end

        # Convert all float parameters to Float16
        temp_f16 = to_float16(temperature)
        top_p_f16 = to_float16(top_p)
        penalty_f16 = to_float16(presence_penalty)
        rep_f16 = to_float16(repetition_penalty)
        min_p_f16 = to_float16(min_p)

        stream = Engine.generate_stream(model, tok, prompt;
            max_tokens=max_tokens, temperature=temp_f16, top_p=top_p_f16,
            top_k=top_k, presence_penalty=penalty_f16, repetition_penalty=rep_f16, min_p=min_p_f16, stop_token=stop_token)

        generated_text = IOBuffer()
        try
            if show_tps
                t0 = time()
                token_count = 0
            end
            for token in stream
                print(io, token)
                flush(io)
                print(generated_text, token)
                if show_tps
                    token_count += 1
                end
            end
            println(io)
            flush(io)
            if show_tps
                elapsed = time() - t0
                tps = elapsed > 0 ? token_count / elapsed : 0.0
                @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
                flush(io)
            end
            return String(take!(generated_text))
        catch e
            if e isa InterruptException
                # Signal the generator to stop if possible
                try
                    close(stream)
                catch
                end
                println(io)
                flush(io)
                if show_tps
                    elapsed = time() - t0
                    tps = elapsed > 0 ? token_count / elapsed : 0.0
                    @printf(io, "[t/s] %.2f tokens/s — %d tokens in %.3fs\n", tps, token_count, elapsed)
                    flush(io)
                end
                return String(take!(generated_text))
            else
                rethrow(e)
            end
        end
elseif chosen_backend == :cpu
 # CPU backend accepts BPETokenizer directly
 if !(tok isa Tokenizer.BPETokenizer)
 error("CPU backend requires a Tokenizer.BPETokenizer. Provided tokenizer is $(typeof(tok)).")
 end

 # Enable BF16 if requested
 if use_bf16
 set_inference_precision!(:bf16)
 @info "BF16 inference enabled for CPU backend"
 else
 set_inference_precision!(:f32)
 end

 # Convert float params to Float32 for CPU backend
 temp_f32 = Float32(temperature)
 top_p_f32 = Float32(top_p)
 penalty_f32 = Float32(presence_penalty)
 rep_f32 = Float32(repetition_penalty)
 min_p_f32 = Float32(min_p)
stop_tokens = stop_token === nothing ? Set{Int}() : Set([stop_token])

    # Dispatch based on model type
    if model isa Gemma4.Gemma4Model
        # Gemma4 CPU generation
        return Gemma4.stream_to_stdout_gemma4_cpu(model, tok, prompt;
            max_tokens=max_tokens, temperature=temp_f32, top_p=top_p_f32, top_k=top_k,
            presence_penalty=penalty_f32, repetition_penalty=rep_f32, min_p=min_p_f32,
            stop_tokens=stop_tokens, show_tps=show_tps, io=io,
            interrupt_check=interrupt_check)
    else
        # Qwen CPU generation
        return ModelCPU.stream_to_stdout_cpu(model, tok, prompt;
            max_tokens=max_tokens, temperature=temp_f32, top_p=top_p_f32, top_k=top_k,
            presence_penalty=penalty_f32, repetition_penalty=rep_f32, min_p=min_p_f32,
            stop_tokens=stop_tokens, show_tps=show_tps, io=io,
            interrupt_check=interrupt_check)
    end
    else
        error("Unsupported backend: $(backend). Use :cpu or :gpu.")
    end
end

include("Chat.jl")
using .Chat

function __init__()
    # Register atexit hook to ensure GPU synchronization and cleanup
    # This helps prevent device lockups on unclean shutdowns
    atexit() do
        try
            oneAPI.synchronize()
        catch
        end
    end
end

end # module
