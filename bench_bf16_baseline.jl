using Inferno

const MODEL_PATH = "/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

println("Loading model...")
model, tok = Inferno.load_model_cpu(MODEL_PATH; use_bf16_weights=false)
