using Inferno
using Printf

# Benchmark: BF16 vs F32 weights on Arrow Lake (Core Ultra 7 265KF)

const PROMPT = "What is 2 + 2 ?"
const MAX_TOKENS = 128

println("=" ^ 60)
println("Arrow Lake BF16 vs F32 Benchmark")
println("=" ^ 60)

# Show CPU features
Inferno.ArrowLake.print_cpu_features()
println()

# --- F32 baseline ---
println("\n--- F32 Baseline (use_bf16_weights=false) ---")
model_f32, tok = Inferno.LoaderCPU.load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"; use_bf16_weights=false)

# Warmup
Inferno.stream_to_stdout(model_f32, tok, "Hello"; max_tokens=5, io=devnull)

# Benchmark
t_f32 = @elapsed result_f32 = Inferno.stream_to_stdout(model_f32, tok, PROMPT; max_tokens=MAX_TOKENS, io=devnull)
tok_count_f32 = length(result_f32)
ms_per_tok_f32 = (t_f32 / tok_count_f32) * 1000
tok_per_s_f32 = tok_count_f32 / t_f32

@printf("F32: %.2f s total, %d tokens, %.2f ms/token, %.1f tok/s\n", t_f32, tok_count_f32, ms_per_tok_f32, tok_per_s_f32)

# --- BF16 ---
println("\n--- BF16 Weights (Arrow Lake optimized) ---")
model_bf16, tok2 = Inferno.LoaderCPU.load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf"; use_bf16_weights=true)

# Warmup
Inferno.stream_to_stdout(model_bf16, tok2, "Hello"; max_tokens=5, io=devnull)

# Benchmark
t_bf16 = @elapsed result_bf16 = Inferno.stream_to_stdout(model_bf16, tok2, PROMPT; max_tokens=MAX_TOKENS, io=devnull)
tok_count_bf16 = length(result_bf16)
ms_per_tok_bf16 = (t_bf16 / tok_count_bf16) * 1000
tok_per_s_bf16 = tok_count_bf16 / t_bf16

@printf("BF16: %.2f s total, %d tokens, %.2f ms/token, %.1f tok/s\n", t_bf16, tok_count_bf16, ms_per_tok_bf16, tok_per_s_bf16)

# --- Summary ---
println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
@printf("F32:  %.2f ms/token, %.1f tok/s\n", ms_per_tok_f32, tok_per_s_f32)
@printf("BF16: %.2f ms/token, %.1f tok/s\n", ms_per_tok_bf16, tok_per_s_bf16)
if ms_per_tok_f32 > 0
    @printf("Speedup: %.2fx\n", ms_per_tok_f32 / ms_per_tok_bf16)
end

# Check correctness
println("\n--- Correctness Check ---")
println("F32 output:  ", result_f32[1:min(80, length(result_f32))])
println("BF16 output: ", result_bf16[1:min(80, length(result_bf16))])
