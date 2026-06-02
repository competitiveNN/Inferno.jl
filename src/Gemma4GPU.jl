module Gemma4GPU

export Gemma4ModelGPU, load_gemma4_gpu, forward_gpu, init_kv_cache_gpu

struct Gemma4ModelGPU
    config::Any
end

struct Gemma4GPUConfig
    hidden_size::Int
    num_layers::Int
    num_heads::Int
    head_dim::Int
    vocab_size::Int
    sliding_window::Int
    max_seq_len::Int
end

function load_gemma4_gpu(path::AbstractString)
    throw(ErrorException("Gemma4 GPU loader is stubbed and not available."))
end

function forward_gpu(model::Gemma4ModelGPU, tokens::Vector{Int}, pos::Int, cache)
    throw(ErrorException("Gemma4 GPU forward is stubbed and not available."))
end

function init_kv_cache_gpu(config::Any)
    throw(ErrorException("Gemma4 GPU KV cache is stubbed and not available."))
end

end

