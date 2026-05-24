module Engine

using ..Model
using ..Tokenizer
using Random

export generate, generate_stream, sample, stream_to_stdout

# Simple sampling with Float32 accumulation for numerical stability
# Matches llama.cpp's approach (cross-entropy-loss.cu:29-33)
function simple_sample(probs::Vector{Float16})
    r = rand(Float32)
    cum = Float32(0.0)
    for (i, p) in enumerate(probs)
        cum += Float32(p)
        if r <= cum
            return i
        end
    end
    return length(probs)
end

function sample(logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40, min_p::Float16=Float16(0.0))
    return cpu_sample(logits, temperature, top_p, top_k, min_p)
end

# CPU sampling implementation with Float32 stability
function cpu_sample(logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40, min_p::Float16=Float16(0.0))
    # Sanitize logits - replace NaN and Inf with -Inf so they're never selected
    @inbounds for i in eachindex(logits)
        if !isfinite(logits[i])
            logits[i] = -Float16(Inf)
        end
    end

    if temperature == Float16(0.0)
        return argmax(logits)
    end

    # Top-k sampling
    if top_k > 0 && top_k < length(logits)
        top_k_indices = partialsortperm(logits, 1:top_k, rev=true)
        for i in eachindex(logits)
            if i ∉ top_k_indices
                logits[i] = -Float16(Inf)
            end
        end
    end

    # Temperature scaling + softmax using Float32 to prevent overflow
    inv_temp = Float32(1.0) / Float32(temperature)
    mx = -Float32(Inf)
    @inbounds for v in logits
        scaled = Float32(v) * inv_temp
        mx = scaled > mx ? scaled : mx
    end
    probs = similar(logits)
    s = Float32(0.0)
    @inbounds for i in eachindex(logits)
        e = exp(Float32(logits[i]) * inv_temp - mx)
        probs[i] = Float16(e)
        s += e
    end
    inv_s = Float32(1.0) / s
    @inbounds probs .*= Float16(inv_s)

    # Top-p (Nucleus) sampling
    if top_p < Float16(1.0)
        n = length(probs)
        k_max = min(n, 1024)
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = Float16(0.0)
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end
        fill!(probs, Float16(0.0))
        renorm = Float32(0.0)
        @inbounds for rank in 1:cutoff
            idx = top_indices[rank]
            e = exp(Float32(logits[idx]) * inv_temp - mx)
            probs[idx] = Float16(e)
            renorm += e
        end
        inv_renorm = Float32(1.0) / renorm
        @inbounds for rank in 1:cutoff
            probs[top_indices[rank]] *= Float16(inv_renorm)
        end
    end

    # Apply minimum probability floor and renormalize
    if min_p > Float16(0.0)
        probs32 = Float32.(probs)
        @inbounds for i in eachindex(probs32)
            probs32[i] = max(probs32[i], Float32(min_p))
        end
        total = sum(probs32)
        if total > 0.0f0
            probs32 ./= total
        end
        probs .= Float16.(probs32)
    end

    return simple_sample(probs)
end

# CPU sampling from already scaled logits
function cpu_sample_from_scaled(scaled_logits::Vector{Float16}, temperature::Float16, top_p::Float16, top_k::Int=40, min_p::Float16=Float16(0.0))
    mx = Float32(maximum(scaled_logits))
    exp_vals = exp.(Float32.(scaled_logits) .- mx)
    probs = Float16.(exp_vals ./ sum(exp_vals))

    if top_k > 0 && top_k < length(scaled_logits)
        top_k_indices = partialsortperm(probs, 1:min(top_k, length(probs)), rev=true)
        for i in eachindex(probs)
            if i ∉ top_k_indices
                probs[i] = Float16(0.0)
            end
        end
        total_prob = sum(probs)
        if total_prob > Float16(0.0)
            probs ./= total_prob
        end
    end

    if top_p < Float16(1.0)
        k_max = min(length(probs), 1024)
        top_indices = partialsortperm(probs, 1:k_max, rev=true)
        cum = Float16(0.0)
        cutoff = k_max
        for (rank, idx) in enumerate(top_indices)
            cum += probs[idx]
            if cum >= top_p
                cutoff = rank
                break
            end
        end
        fill!(probs, Float16(0.0))
        for rank in 1:cutoff
            probs[top_indices[rank]] = exp_vals[top_indices[rank]]
        end
        probs ./= sum(probs)
    end

    # Apply minimum probability floor and renormalize
    if min_p > Float16(0.0)
        # Work in Float32 for stability
        probs32 = Float32.(probs)
        @inbounds for i in eachindex(probs32)
            probs32[i] = max(probs32[i], Float32(min_p))
        end
        total = sum(probs32)
        if total > 0.0f0
            probs32 ./= total
        end
        probs .= Float16.(probs32)
    end

    return simple_sample(probs)
end

# mask_and_sample: avoids returning PAD-like tokens when streaming
function mask_and_sample(logits::AbstractVector{Float16}, pad_ids::Vector{Int}, temperature::Float16, top_p::Float16, top_k::Int=40; max_attempts::Int=5, min_p::Float16=Float16(0.0))
    logits_copy = logits isa Vector{Float16} ? copy(logits) : Vector{Float16}(logits)

    if !isempty(pad_ids)
        @inbounds for pid in pad_ids
            if 1 <= pid <= length(logits_copy)
                logits_copy[pid] = -Float16(Inf)
            end
        end
    end

    for attempt in 1:max_attempts
        cand = sample(logits_copy, temperature, top_p, top_k, min_p)
        if !isempty(pad_ids) && cand <= length(logits_copy) && logits_copy[cand] <= -1e8
            logits_copy[cand] = -Float16(Inf)
            continue
        end
        return cand
    end

    return argmax(logits_copy)
end

# Apply presence penalty to logits based on token counts
function apply_presence_penalty!(logits::AbstractVector{Float16}, token_counts::Dict{Int,Int}, penalty::Float16)
    if penalty == Float16(0.0)
        return logits
    end
    for (tokid, _cnt) in token_counts
        if 1 <= tokid <= length(logits)
            logits[tokid] -= penalty
        end
    end
    return logits
end

# Apply repetition penalty (different from presence - applies to all occurrences)
function apply_repetition_penalty!(logits::AbstractVector{Float16}, token_counts::Dict{Int,Int}, penalty::Float16)
    if penalty == Float16(1.0)
        return logits
    end
    for (tokid, cnt) in token_counts
        if 1 <= tokid <= length(logits)
            # Repetition penalty: divide logits for tokens that have appeared
            if logits[tokid] > Float16(0.0)
                logits[tokid] = logits[tokid] / penalty
            else
                logits[tokid] = logits[tokid] * penalty
            end
        end
    end
    return logits
end

# Qwen2.5/Qwen3.5 recommended defaults (from generation_config.json):
# - temperature: 0.0 (use > 0 for sampling)
# - top_p: 0.95
# - top_k: 0 (disabled, use all tokens in top-p)
# - repetition_penalty: 1.0 (use 1.1-1.2 for long texts)
# - gc_interval: 0 (disabled by default to avoid stuttering, set > 0 to enable periodic GC)

function generate_stream(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=512,
    temperature::Float16=Float16(0.7),
    top_p::Float16=Float16(0.95),
    top_k::Int=0,
    presence_penalty::Float16=Float16(0.0),
    repetition_penalty::Float16=Float16(1.0),
    min_p::Float16=Float16(0.0),
    stop_token::Union{Int,Nothing}=nothing,
    gc_interval::Int=0)

    tokens = Tokenizer.encode(tok, prompt)
    if isempty(tokens)
        return Channel{String}(0) do chan
            close(chan)
        end
    end

    token_counts = Dict{Int,Int}()
    for t in tokens
        token_counts[t] = get(token_counts, t, 0) + 1
    end

    kv_caches = Vector{Model.KVCache}()
    return Channel{String}(32) do chan
        try
            devs = collect(oneAPI.devices())
            if !isempty(devs)
                first_weight = model.layers[1].mlp.gate_weight
                weight_device = device(first_weight)
                oneAPI.device!(weight_device)
            end

            Model.reset_states!(model)
            kv_caches = [Model.init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

            logits = Model.forward!(model, tokens, 0, kv_caches; gc_interval=gc_interval)
            curr_pos = length(tokens)

            logits_vec = vec(collect(logits[:, end]))
            apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
            apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
            last_token = mask_and_sample(logits_vec, [tok.eos_id], temperature, top_p, top_k; min_p=min_p)

            token_str = Tokenizer.decode(tok, [last_token])
            put!(chan, token_str)

            token_counts[last_token] = get(token_counts, last_token, 0) + 1

            for step in 1:(max_tokens-1)
                if last_token == (stop_token === nothing ? tok.eos_id : stop_token)
                    break
                end

                logits = Model.forward!(model, [last_token], curr_pos, kv_caches; gc_interval=gc_interval)
                curr_pos += 1

                logits_vec = vec(collect(logits[:, 1]))
                apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
                apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
                last_token = mask_and_sample(logits_vec, [tok.eos_id], temperature, top_p, top_k; min_p=min_p)

                token_str = Tokenizer.decode(tok, [last_token])
                put!(chan, token_str)

                token_counts[last_token] = get(token_counts, last_token, 0) + 1

                # Optional GC during generation (disabled by default)
                if gc_interval > 0 && step % gc_interval == 0
                    GC.gc(false)
                end
            end

            oneAPI.synchronize()

        catch e
            if !(e isa InterruptException) && !(e isa InvalidStateException)
                @error "ERROR during generation stream" exception=(e, catch_backtrace())
            end
        finally
            try
                if !isempty(kv_caches)
                    Model.free_all_kv_caches!(kv_caches)
                end
            catch
            end
            try
                GC.gc(true)
            catch
            end
            try
                close(chan)
            catch
            end
        end
    end
end

# =======================================================================
# generate_stream for Gemma4 GPU models
# =======================================================================
using ..Gemma4
using ..Gemma4GPU
using ..Gemma4GPUKernels
using ..Qwen35GPU

function generate_stream(model::Gemma4GPU.Gemma4ModelGPU, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=512,
    temperature::Float16=Float16(0.7),
    top_p::Float16=Float16(0.95),
    top_k::Int=0,
    presence_penalty::Float16=Float16(0.0),
    repetition_penalty::Float16=Float16(1.0),
    min_p::Float16=Float16(0.0),
    stop_token::Union{Int,Nothing}=nothing,
    gc_interval::Int=0)

    # Gemma4 chat format
    end_turn = 107  # <turn|> (1-indexed)
    chat_prompt = "<|turn>user\n$(prompt)<turn|><|turn>model\n"
    
    # Encode prompt
    tokens = Tokenizer.encode(tok, chat_prompt)
    
    # Add BOS token (3 in 1-indexed)
    if isempty(tokens) || tokens[1] != 3
        pushfirst!(tokens, 3)
    end
    
    if isempty(tokens)
        return Channel{String}(0) do chan
            close(chan)
        end
    end

    token_counts = Dict{Int,Int}()
    for t in tokens
        token_counts[t] = get(token_counts, t, 0) + 1
    end

    return Channel{String}(32) do chan
        try
            # Initialize KV cache for Gemma4 GPU
            cache = produce_empty_kv_cache_gemma4(model)
            
            # Process prompt tokens
            logits = Gemma4GPU.forward_gpu(model, tokens, 0, cache)
            curr_pos = length(tokens)

            # Sample first token
            logits_vec = vec(collect(logits[:, end]))
            apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
            apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
            last_token = mask_and_sample(logits_vec, [tok.eos_id, end_turn], temperature, top_p, top_k; min_p=min_p)

            token_str = Tokenizer.decode(tok, [last_token])
            put!(chan, token_str)
            token_counts[last_token] = get(token_counts, last_token, 0) + 1

            # Generate loop
            for step in 1:(max_tokens-1)
                stop_id = stop_token === nothing ? tok.eos_id : stop_token
                if last_token == stop_id || last_token == end_turn
                    break
                end

                logits = Gemma4GPU.forward_gpu(model, [last_token], curr_pos, cache)
                curr_pos += 1

                logits_vec = vec(collect(logits[:, 1]))
                apply_presence_penalty!(logits_vec, token_counts, presence_penalty)
                apply_repetition_penalty!(logits_vec, token_counts, repetition_penalty)
                last_token = mask_and_sample(logits_vec, [tok.eos_id, end_turn], temperature, top_p, top_k; min_p=min_p)

                token_str = Tokenizer.decode(tok, [last_token])
                put!(chan, token_str)
                token_counts[last_token] = get(token_counts, last_token, 0) + 1
            end

            oneAPI.synchronize()

        catch e
            if !(e isa InterruptException)
                @error "ERROR during Gemma4 GPU stream" exception=(e, catch_backtrace())
            end
        finally
            try
                GC.gc(true)
            catch
            end
            try
                close(chan)
            catch
            end
        end
    end
end

# Helper to create empty KV cache for Gemma4 GPU
function produce_empty_kv_cache_gemma4(model::Gemma4GPU.Gemma4ModelGPU)
    # Gemma4 uses sliding window for most layers, shared KV for some
    config = model.config
    sw = config.sliding_window
    cache = Gemma4GPU.init_kv_cache_gemma4(config)
    return cache
end

function generate(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=512,
    temperature::Float16=Float16(0.7),
    top_p::Float16=Float16(0.95),
    top_k::Int=0,
    presence_penalty::Float16=Float16(0.0),
    repetition_penalty::Float16=Float16(1.0),
    min_p::Float16=Float16(0.0),
    stop_token::Union{Int,Nothing}=nothing,
    gc_interval::Int=0)

    stream = generate_stream(model, tok, prompt; max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, min_p=min_p, stop_token=stop_token, gc_interval=gc_interval)
    res = String[]
    for token_str in stream
        push!(res, token_str)
    end
    return join(res)
end

function stream_to_stdout(model::Model.QwenModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=512,
    temperature::Float16=Float16(0.7),
    top_p::Float16=Float16(0.95),
    top_k::Int=0,
    presence_penalty::Float16=Float16(0.0),
    repetition_penalty::Float16=Float16(1.0),
    min_p::Float16=Float16(0.0),
    gc_interval::Int=0)

    stream = generate_stream(model, tok, prompt; max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, min_p=min_p, gc_interval=gc_interval)
    for token_str in stream
        print(token_str)
    end
    println()
end


# ------------------------------------------------------------
# GPU dispatch for Qwen35GPU (Intel Arc)
# ------------------------------------------------------------
function generate_stream(model::Qwen35GPU.Qwen35GPUModel, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=512,
    temperature::Float16=Float16(0.7),
    top_p::Float16=Float16(0.95),
    top_k::Int=0,
    presence_penalty::Float16=Float16(0.0),
    repetition_penalty::Float16=Float16(1.0),
    min_p::Float16=Float16(0.0),
    stop_token::Union{Int,Nothing}=nothing,
    gc_interval::Int=0)
    return Qwen35GPU.generate_stream(model, tok, prompt;
        max_tokens=max_tokens,
        temperature=Float32(temperature),
        top_p=Float32(top_p),
        top_k=top_k)
end

end # module
