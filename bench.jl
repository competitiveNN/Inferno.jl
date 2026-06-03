using Inferno
model, tok = Inferno.load_model_cpu("/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
toks = Inferno.encode(tok, "<|turn>user
What is 2 + 2 ?<turn|><|turn>model
")
Inferno.reset_states_cpu!(model)
t0 = time()
function run()
 local out = ""
 for tok_str in Inferno.generate_stream_cpu(model, toks, t -> Inferno.decode(tok, t); max_tokens=64)
  out *= tok_str
 end
 return out
end
text = run()
t1 = time()
count = length(Inferno.encode(tok, text))
println("tps=", round(count/(t1-t0), digits=1))
println("text=", text)
