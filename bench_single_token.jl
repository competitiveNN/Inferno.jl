using Inferno
using Inferno.ModelCPU: init_kv_cache_cpu, forward_cpu!, reset_states_cpu!

const MODEL_PATH = "/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

model, tok = Inferno.load_model_cpu(MODEL_PATH)
caches = [init_kv_cache_cpu(model.config, 512) for i in 1:model.config.num_hidden_layers]

toks = [1]

# Warmup
forward_cpu!(model, toks, 1, caches)
reset_states_cpu!(model)

# Benchmark
n = 10
total = 0.0
for i in 1:n
    t0 = time()
    forward_cpu!(model, toks, 1, caches)
    t1 = time()
    total += (t1-t0)
end

avg_ms = total/n*1000
println("Average forward_cpu! time: $(round(avg_ms, digits=2))ms")
println("Tokens/sec: $(round(1/(total/n), digits=2))")
