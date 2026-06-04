module BenchAB

using Inferno
using Inferno.Tokenizer: encode
using LinearAlgebra

const PATH = "/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

function try_mode(label, use_bf16)
  println("=== $label ===")
  m, tok = Inferno.load_model_cpu(PATH; keep_quantized=false, use_bf16_weights=use_bf16)
  ctx = encode(tok, "What is 2 + 2 ?")
  stop = encode(tok, "aa")
  s = @elapsed Inferno.generate_stream_cpu(m, ctx, 1; temperature=1.0, stop_tokens=stop, max_tokens=16, show_tps=false)
  println("elapsed: ", s, "s  approx tps: ", round(16/s, digits=2))
end

try_mode("F32 baseline", false)
try_mode("BF16 weights", true)

end

BenchAB.try_mode("F32 baseline", false)
BenchAB.try_mode("BF16 weights", true)
