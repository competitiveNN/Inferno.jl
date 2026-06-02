using Inferno
m, t = Inferno.load_model_cpu("/var/home/fra/data/models/gguf/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
Inferno.ModelCPU.init_layer_counters!(length(m.layers))
Inferno.ModelCPU.reset_layer_counters!()

# Fixed 5-token trace (fast replay)
toks = Inferno.encode(t, "<|turn>user\nhi<turn|><|turn>model\n")
Inferno.reset_states_cpu!(m)
function run()
    local n = 0
    for _ in Inferno.generate_stream_cpu(m, toks, x -> Inferno.decode(t, x))
        n += 1
        if n >= 5
            break
        end
    end
    return n
end
run()
println(Inferno.ModelCPU.format_layer_summary())
