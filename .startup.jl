# MODEL_DIR = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf"
# MODEL_DIR = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q5_K_XL.gguf"
# MODEL_DIR = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q6_K_XL.gguf"
MODEL_DIR = "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"
# MODEL_DIR = "Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
# MODEL_DIR = "Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-TQ1_0.gguf"
# MODEL_DIR = "Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
# MODEL_DIR = "Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf"
# MODEL_DIR = "Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
# MODEL_DIR = "gemma-4-E2B-it/model.safetensors"
# MODEL_DIR = "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
# MODEL_DIR = "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf"
# MODEL_DIR = "gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf"
ENV["MODEL_PATH"] = joinpath("/home/fra/data/models/gguf/", MODEL_DIR)
global const MODEL_PATH = ENV["MODEL_PATH"]
using Inferno
#model, bpetok = load_model(MODEL_PATH, backend=:cpu)
#stream_to_stdout(model, bpetok, "2 + 2 =" , top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0, temperature=0.2, max_tokens=120, backend=:cpu, show_tps=true);
