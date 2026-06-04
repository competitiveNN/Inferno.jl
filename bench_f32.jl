module BenchF32

using Inferno
using Inferno.LoaderCPU: load_model_cpu
using Inferno.ModelCPU: QwenModelCPU, init_kv_cache_cpu, forward_cpu!
using BenchmarkTools

const MODEL_PATH = "/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

function bench()
  println("Loading...")
  model = load_model_cpu(MODEL_PATH; use_bf16_weights=false)
  caches = init_kv_cache_cpu(model)
  x = rand(Float32, model.config.hidden_size)

  t = @time forward_cpu!(model, x, caches, 0, 1.0f0)
  println("Single-token forward: ", t)
  println("Tokens/sec: ", 1.0/t)
end

end

BenchF32.bench()
