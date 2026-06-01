using Inferno
using Printf

# Benchmark: Arrow Lake optimized F32 inference
const PROMPT = "What is 2 + 2 ?"
const MAX_TOKENS = 128

println("=" ^ 60)
println("Arrow Lake F32 Inference Benchmark")
println("=" ^ 60)
Inferno.ArrowLake.print_cpu_features()

# Load model (BF16 disabled by default)
println("\nLoading model (F32 weights)...")
model, tok = Inferno.LoaderCPU.load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Warmup
Inferno.stream_to_stdout(model, tok, "Hello"; max_tokens=5, io=devnull)

# Run 3 trials
times = Float64[]
for trial in 1:3
    t = @elapsed result = Inferno.stream_to_stdout(model, tok, PROMPT; max_tokens=MAX_TOKENS, io=devnull)
    n_tok = length(result)
    ms = (t / n_tok) * 1000
    tps = n_tok / t
    push!(times, ms)
    @printf("Trial %d: %.2f ms/token, %.1f tok/s (%d tokens)\n", trial, ms, tps, n_tok)
end

@printf("\nAverage: %.2f ms/token, %.1f tok/s\n", mean(times), 1000.0/mean(times))

# Verify output
result = Inferno.stream_to_stdout(model, tok, "The capital of France"; max_tokens=30, io=devnull)
println("\nCorrectness: ", result[1:min(100, length(result))])

using Statistics
