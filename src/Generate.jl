"""
High-level generation API for CPU backend.
Provides simple text-in/text-out interface.
"""
module Generate

using ..GGUF
using ..LoaderCPU
using ..ModelCPU
using ..Tokenizer

export generate_text, chat, SimpleTokenizer

"""
Simple tokenizer using GGUF token list.
For production use, implement proper BPE tokenization.
"""
mutable struct SimpleTokenizer
    tokens::Vector{String}
    token_to_id::Dict{String, Int}
    bos_id::Int
    eos_id::Int
    pad_id::Int
end

function SimpleTokenizer(file::GGUF.GGUFFile)
    tokens = file.metadata["tokenizer.ggml.tokens"]
    token_to_id = Dict{String, Int}()
    for (i, t) in enumerate(tokens)
        token_to_id[t] = i - 1  # 0-indexed
    end
    
    # Get special token IDs
    bos_id = get(file.metadata, "tokenizer.ggml.bos_token_id", 1)
    eos_id = get(file.metadata, "tokenizer.ggml.eos_token_id", 151645)
    pad_id = get(file.metadata, "tokenizer.ggml.padding_token_id", 151643)
    
    return SimpleTokenizer(tokens, token_to_id, bos_id, eos_id, pad_id)
end

function encode(tok::SimpleTokenizer, text::String)
    # Very simple tokenization - try whole string first, then character by character
    # This is NOT a proper BPE tokenizer - just for basic functionality
    
    token_ids = Int[]
    
    # Add BOS if text doesn't start with special format
    # push!(token_ids, tok.bos_id)
    
    # Try to find tokens
    remaining = text
    while !isempty(remaining)
        found = false
        
        # Try longest match first
        for len in length(remaining):-1:1
            candidate = SubString(remaining, 1, len)
            
            # Try with and without space prefix
            for prefix in ["", "Ġ"]
                key = prefix * candidate
                if haskey(tok.token_to_id, key)
                    push!(token_ids, tok.token_to_id[key])
                    remaining = len < length(remaining) ? SubString(remaining, len + 1) : ""
                    found = true
                    break
                end
            end
            found && break
        end
        
        if !found
            # Skip unknown character
            if length(remaining) > 1
                remaining = SubString(remaining, 2)
            else
                break
            end
        end
    end
    
    return token_ids
end

function decode(tok::SimpleTokenizer, ids::Vector{Int})
    parts = String[]
    for id in ids
        if 1 <= id + 1 <= length(tok.tokens)
            t = tok.tokens[id + 1]
            # Handle GPT-2 style space prefix
            t = replace(t, "Ġ" => " ")
            push!(parts, t)
        end
    end
    return join(parts)
end

function encode(tok::Tokenizer.BPETokenizer, text::String)
    return Tokenizer.encode(tok, text)
end

function decode(tok::Tokenizer.BPETokenizer, ids::Vector{Int})
    return Tokenizer.decode(tok, ids)
end

"""
    generate_text(model, tokenizer, prompt; kwargs...)

Generate text from a prompt using the CPU model.

# Example
```julia
model, file = load_model_cpu("model.gguf")
tok = SimpleTokenizer(file)

output = generate_text(model, tok, "What is machine learning?")
println(output)
```

# Arguments
- `model`: The loaded QwenModelCPU
- `tokenizer`: SimpleTokenizer instance
- `prompt`: Text prompt

# Keyword Arguments
- `max_tokens`: Maximum tokens to generate (default: 256)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling threshold (default: 0.9)
- `top_k`: Top-k filtering, 0 to disable (default: 40)
- `repetition_penalty`: Penalty for repeated tokens (default: 1.1)
"""
function generate_text(model::ModelCPU.QwenModelCPU, tok::SimpleTokenizer, prompt::String;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    stop_tokens::Set{Int}=Set{Int}())
    
    # Add EOS to stop tokens
    push!(stop_tokens, tok.eos_id)
    
    # Tokenize prompt
    prompt_tokens = encode(tok, prompt)
    
    if isempty(prompt_tokens)
        return ""
    end
    
    # Create decode function
    decode_fn = (ids) -> decode(tok, ids)
    
    # Generate
    result = ModelCPU.stream_to_stdout_cpu(
        model,
        prompt_tokens,
        decode_fn;
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_tokens=stop_tokens
    )
    
    return result
end

function generate_text(model::ModelCPU.QwenModelCPU, tok::Tokenizer.BPETokenizer, prompt::String;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    stop_tokens::Set{Int}=Set{Int}())

    # Add EOS to stop tokens (eos_id is already 1-indexed)
    push!(stop_tokens, tok.eos_id)

    # Tokenize prompt (returns 1-indexed IDs)
    prompt_tokens = encode(tok, prompt)

    if isempty(prompt_tokens)
        return ""
    end

    # Create decode function
    decode_fn = (ids) -> decode(tok, ids)

    # Generate
    result = ModelCPU.stream_to_stdout_cpu(
        model,
        prompt_tokens,
        decode_fn;
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_tokens=stop_tokens
    )

    return result
end

"""
    generate_text(model, tokenizer, prompt, state::ModelCPU.GenerationState; kwargs...)

Generate text using pre-allocated KV caches from `GenerationState`.
This avoids the ~800MB allocation per call by reusing caches.

# Example
```julia
state = ModelCPU.create_generation_state(model; max_context=8192)
output1 = generate_text(model, tok, "Hello", state; max_tokens=20)
# Continue with same state
output2 = generate_text(model, tok, "World", state; max_tokens=20)
```
"""
function generate_text(model::ModelCPU.QwenModelCPU, tok::SimpleTokenizer, prompt::String,
    state::ModelCPU.GenerationState;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    stop_tokens::Set{Int}=Set{Int}())

    push!(stop_tokens, tok.eos_id)
    prompt_tokens = encode(tok, prompt)
    if isempty(prompt_tokens)
        return ""
    end
    return ModelCPU.generate_with_cache(model, tok, prompt, state;
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, stop_tokens=stop_tokens)
end

function generate_text(model::ModelCPU.QwenModelCPU, tok::Tokenizer.BPETokenizer, prompt::String,
    state::ModelCPU.GenerationState;
    max_tokens::Int=256,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0,
    stop_tokens::Set{Int}=Set{Int}())

    push!(stop_tokens, tok.eos_id)
    prompt_tokens = encode(tok, prompt)
    if isempty(prompt_tokens)
        return ""
    end
    return ModelCPU.generate_with_cache(model, tok, prompt, state;
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, stop_tokens=stop_tokens)
end

"""
    chat(model, tokenizer, messages; kwargs...)

Chat-style generation with message history.

# Example
```julia
model, file = load_model_cpu("model.gguf")
tok = SimpleTokenizer(file)

messages = [
    ("system", "You are a helpful assistant."),
    ("user", "What is 2+2?")
]

response = chat(model, tok, messages)
println(response)
```
"""
function chat(model::ModelCPU.QwenModelCPU, tok::SimpleTokenizer, messages::Vector{Tuple{String,String}};
    max_tokens::Int=512,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0)
    
    # Format messages for Qwen chat template (correct <|im_start|> format)
    parts = String[]
    for (role, content) in messages
        if role == "system"
            push!(parts, "<|im_start|>system\n$(content)<|im_end|>")
        elseif role == "user"
            push!(parts, "<|im_start|>user\n$(content)<|im_end|>")
        elseif role == "assistant"
            push!(parts, "<|im_start|>assistant\n$(content)<|im_end|>")
        end
    end
    push!(parts, "<|im_start|>assistant\n")
    
    prompt = join(parts, "\n")
    
    # Add im_end and eos as stop sequences
    stop_tokens = Set{Int}()
    # Try to find <|im_end|> token
    for (i, t) in enumerate(tok.tokens)
        if occursin("<|im_end|>", t)
            push!(stop_tokens, i - 1)
        end
    end
    push!(stop_tokens, tok.eos_id)
    
    return generate_text(model, tok, prompt;
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_tokens=stop_tokens
    )
end

function chat(model::ModelCPU.QwenModelCPU, tok::Tokenizer.BPETokenizer, messages::Vector{Tuple{String,String}};
    max_tokens::Int=512,
    temperature::Float32=0.7f0,
    top_p::Float32=0.9f0,
    top_k::Int=40,
    repetition_penalty::Float32=1.1f0)

    # Format messages for Qwen chat template (correct <|im_start|> format)
    parts = String[]
    for (role, content) in messages
        if role == "system"
            push!(parts, "<|im_start|>system\n$(content)<|im_end|>")
        elseif role == "user"
            push!(parts, "<|im_start|>user\n$(content)<|im_end|>")
        elseif role == "assistant"
            push!(parts, "<|im_start|>assistant\n$(content)<|im_end|>")
        end
    end
    push!(parts, "<|im_start|>assistant\n")

    prompt = join(parts, "\n")

    # Add im_end and eos as stop sequences (convert 1-indexed to 0-indexed)
    stop_tokens = Set{Int}()
    # Try to find <|im_end|> token
    for (i, t) in enumerate(tok.id_to_token)
        if occursin("<|im_end|>", t)
            push!(stop_tokens, i - 1)  # Convert 1-indexed to 0-indexed
        end
    end
    push!(stop_tokens, tok.eos_id - 1)  # Convert 1-indexed to 0-indexed
    
    return generate_text(model, tok, prompt;
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_tokens=stop_tokens
    )
end

end # module
