using Inferno
using Printf
using LinearAlgebra

function run_profile()
    model, tok = Inferno.LoaderCPU.load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")
    
    # Warmup
    Inferno.stream_to_stdout(model, tok, "Hello"; max_tokens=3, io=devnull)
    
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for layer in model.layers]
    Inferno.ModelCPU.reset_states_cpu!(model)
    prompt_tokens = Inferno.Tokenizer.encode(tok, "What is 2 + 2")
    logits = Inferno.ModelCPU.forward_cpu!(model, prompt_tokens, 0, caches)
    pos = length(prompt_tokens)
    token = argmax(logits[:, end])
    
    ssm_time = 0.0
    attn_time = 0.0
    lm_head_time = 0.0
    final_norm_time = 0.0
    
    for _ in 1:20
        for (i, layer) in enumerate(model.layers)
            t_start = time_ns()
            # We need to call forward directly to time individual components
            # Instead, just time whole layers
            x = model.embed[:, token+1]
        end
        
        # Just do a forward pass and time lm_head separately
        t0 = time_ns()
        Inferno.ModelCPU.rmsnorm_cpu!(model.final_norm_buf, model.embed[:, token+1], model.final_norm)
        final_norm_time += time_ns() - t0
        
        t0 = time_ns()
        Inferno.ModelCPU.lm_head_project!(model.lm_head_buf, model.lm_head, model.final_norm_buf)
        lm_head_time += time_ns() - t0
        
        token = argmax(model.lm_head_buf)
        pos += 1
    end
    
    # Time SSM vs Attention by doing a full forward pass
    # We'll do 20 forward passes and time each layer
    ssm_times = Float64[]
    attn_times = Float64[]
    mlp_times = Float64[]
    
    Inferno.ModelCPU.reset_states_cpu!(model)
    caches = [Inferno.ModelCPU.init_kv_cache_cpu(model.config) for _ in model.layers]
    
    for _ in 1:20
        x = model.embed[:, token+1]
        for (i, layer) in enumerate(model.layers)
            t_layer = @elapsed layer(x, pos, model.rope, caches[i])
            if layer.is_ssm
                push!(ssm_times, t_layer * 1000)
            else
                push!(attn_times, t_layer * 1000)
            end
        end
        token = argmax(model.lm_head_buf)
        pos += 1
    end
    
    lm_ms = lm_head_time / 20 / 1e6
    norm_ms = final_norm_time / 20 / 1e6
    
    println("\nPer-token breakdown (20 token average):")
    @printf("  SSM layers (18):  %6.2f ms  (%5.1f%%)\n", mean(ssm_times)*18, 100*mean(ssm_times)*18/(mean(ssm_times)*18+mean(attn_times)*6+lm_ms+norm_ms))
    @printf("  Attn layers (6):  %6.2f ms  (%5.1f%%)\n", mean(attn_times)*6, 100*mean(attn_times)*6/(mean(ssm_times)*18+mean(attn_times)*6+lm_ms+norm_ms))
    @printf("  Final norm:       %6.2f ms  (%5.1f%%)\n", norm_ms, 100*norm_ms/(mean(ssm_times)*18+mean(attn_times)*6+lm_ms+norm_ms))
    @printf("  LM head:          %6.2f ms  (%5.1f%%)\n", lm_ms, 100*lm_ms/(mean(ssm_times)*18+mean(attn_times)*6+lm_ms+norm_ms))
    total_ms = mean(ssm_times)*18 + mean(attn_times)*6 + lm_ms + norm_ms
    @printf("  Total:            %6.2f ms  (%.1f tok/s)\n", total_ms, 1000/total_ms)
end

using Statistics
run_profile()
