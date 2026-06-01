using Inferno, LinearAlgebra
using Printf, Dates, Statistics
using Inferno.Tokenizer
using Inferno.ModelCPU: init_kv_cache_cpu, forward_cpu!

# MODEL_PATH = get(ENV, "INFERNO_MODEL", "test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
MODEL_PATH = "test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"
WARMUP_TOKENS = 10
BENCHMARK_TOKENS = 100

struct BenchResult
    prompt_tokens::Int
    gen_tokens::Int
    prefill_ms::Float64
    decode_ms::Float64
    tps::Float64
end

function benchmark(model, tok, prompt; n_tokens=100)
    tokens = Tokenizer.encode(tok, prompt)
    caches = [init_kv_cache_cpu(model.config) for _ in 1:model.config.num_hidden_layers]
    
    # Prefill
    t0 = time_ns()
    logits = forward_cpu!(model, tokens, 0, caches)
    prefill_ns = time_ns() - t0
    
    # Decode
    pos = length(tokens)
    t1 = time_ns()
    for i in 1:n_tokens
        token = argmax(logits[:, end])
        logits = forward_cpu!(model, [token], pos, caches)
        pos += 1
    end
    decode_ns = time_ns() - t1
    
    prefill_ms = prefill_ns / 1e6
    decode_ms = decode_ns / 1e6
    tps = n_tokens / (decode_ms / 1000)
    
    return BenchResult(length(tokens), n_tokens, prefill_ms, decode_ms, tps)
end

function main()
    println("Loading model...")
    model, tok = load_model(MODEL_PATH; backend=:cpu)
    println("Threads: Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    
    # Warmup
    println("Warming up...")
    benchmark(model, tok, "Hi"; n_tokens=10)
    
    # Benchmark
    println("\n=== BENCHMARK (100 tokens) ===")
    r = benchmark(model, tok, "What is 2+2?"; n_tokens=100)
    
    println("Prompt tokens: $(r.prompt_tokens)")
    println("Prefill: $(round(r.prefill_ms, digits=1))ms ($(round(r.prompt_tokens/(r.prefill_ms/1000), digits=1)) t/s)")
    println("Decode: $(round(r.decode_ms/r.gen_tokens, digits=2))ms/token")
    println("Throughput: $(round(r.tps, digits=1)) tok/s")
    
    if r.tps >= 50
        println("✓ Target: $(round(r.tps, digits=1)) ≥ 50 t/s")
    else
        println("✗ Target: $(round(r.tps, digits=1)) < 50 t/s (gap: $(round(50-r.tps, digits=1)))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
