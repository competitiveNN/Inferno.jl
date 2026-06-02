using Inferno, Profile

model, tok = Inferno.load_model_cpu("/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
tokens = Inferno.encode(tok, "<|turn>user\n2+2?<turn|><|turn>model\n")
Inferno.reset_states_cpu!(model)

Profile.clear()
for _ in 1:5
    Inferno.generate_stream_cpu(model, tokens, t -> Inferno.decode(tok, t); max_tokens=16)
end

Profile.print(format=:flat, minperc=2.0, maxdepth=8)
