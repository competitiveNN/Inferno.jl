#!/usr/bin/env julia
# benchmark/gpu_benchmark.jl - End-to-end GPU generation benchmark for Qwen35GPU (Intel Arc)
# Usage: INFERNO_MODEL=models/Qwen3.5-0.8B-v3_q4_k_m.gguf julia --project=. benchmark/gpu_benchmark.jl

using Printf, Statistics

const MODEL_PATH = get(ENV, "INFERNO_MODEL", "models/Qwen3.5-0.8B-v3_q4_k_m.gguf")
const WARMUP = 8
const BENCH_TOKENS = 256
const PROMPT = "The quick brown fox"

function run_benchmark()
    println("[GPU] Loading model: $MODEL_PATH")
    model, tok = Inferno.load_qwen35_gpu(MODEL_PATH)

    println("[GPU] Warming up ($WARMUP tokens)...")
    Inferno.generate(model, tok, PROMPT; max_tokens=WARMUP)

    println("[GPU] Running $BENCH_TOKENS token generation...")
    t0 = time_ns()
    output = Inferno.generate(model, tok, PROMPT; max_tokens=BENCH_TOKENS)
    t1 = time_ns()
    dt = (t1 - t0) / 1e9
    tps = BENCH_TOKENS / dt
    println("[GPU] Generated $BENCH_TOKENS tokens in $(round(dt, digits=3))s => $(round(tps, digits=2)) tok/s")
    return tps
end

run_benchmark()