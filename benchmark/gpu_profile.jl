#!/usr/bin/env julia
# benchmark/gpu_profile.jl - GPU generation profiling for Qwen35GPU (Intel Arc)
# Usage: INFERNO_MODEL=models/Qwen3.5-0.8B-v3_q4_k_m.gguf julia --project=. benchmark/gpu_profile.jl

using Printf

const MODEL_PATH = get(ENV, "INFERNO_MODEL", "models/Qwen3.5-0.8B-v3_q4_k_m.gguf")
const WARMUP = 8
const BENCH_TOKENS=256 
const PROMPT = "The quick brown fox jumps over the lazy dog. This is a test of GPU inference performance."

function run_profile()
    println("="^70)
    println("  GPU Generation Profiling - Qwen3.5-0.8B (Intel Arc)")
    println("  Model : ", MODEL_PATH)
    println("  Prompt: ", PROMPT)
    println("="^70)
    
    # Load model
    println("\n[1] Loading model...")
    model, tok = Inferno.load_qwen35_gpu(MODEL_PATH)
    println("    Model loaded successfully")
    
    # Warmup
    println("\n[2] Warmup ($WARMUP tokens)...")
    Inferno.generate(model, tok, PROMPT; max_tokens=WARMUP)
    println("    Warmup complete")
    
    # Measure end-to-end generation
    println("\n[3] Running $BENCH_TOKENS token generation...")
    t0 = time_ns()
    output = Inferno.generate(model, tok, PROMPT; max_tokens=BENCH_TOKENS)
    t1 = time_ns()
    
    total_dt = (t1 - t0) / 1e9
    tps = BENCH_TOKENS / total_dt
    ms_per_tok = (total_dt / BENCH_TOKENS) * 1000
    
    # Results
    println("\n" * "="^70)
    println("  Results")
    println("="^70)
    println(@sprintf("  Generated:     %d tokens", BENCH_TOKENS))
    println(@sprintf("  Total time:    %.3f seconds", total_dt))
    println(@sprintf("  Throughput:    %.2f tok/s", tps))
    println(@sprintf("  Latency:       %.2f ms/tok", ms_per_tok))
    println(@sprintf("  Output length:   %d chars", length(output)))
    
    # Recommendations based on throughput
    println("\n" * "="^70)
    println("  Recommendations")
    println("="^70)
    if tps < 50
        println("  Throughput < 50 tok/s. Check:")
        println("  1. GPU driver: ensure zeInit succeeds")
        println("  2. BLAS dispatch: verify mul! -> oneMKL")
        println("  3. GPU kernels: all operations should be GPU-native")
    elseif tps < 100
        println("  Throughput < 100 tok/s. Consider:")
        println("  1. Batch processing (process 4-8 tokens at once)")
        println("  2. Flash Attention for longer sequences")
        println("  3. MLP fusion (gate + up + SiLU + down projection)")
    elseif tps < 140
        println("  Throughput < 140 tok/s (target). Options:")
        println("  1. Batch processing for higher throughput")
        println("  2. GPU-side top-k sampling (avoids 608KB CPU transfer)")
        println("  3. Profile per-layer timing to find remaining bottleneck")
    else
        println("  Throughput >= 140 tok/s. Target reached!")
    end
    
    return tps
end

run_profile()
