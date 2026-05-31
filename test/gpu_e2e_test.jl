# Minimal GPU e2e test for Qwen3.5
# Run: julia --project=. test/gpu_e2e_test.jl <model.gguf>

using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using LinearAlgebra
using Printf

# ── GPU matmul kernel ──
@kernel function matmul_vec8!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    if i <= n
        T = eltype(y)
        acc = zero(T)
        j = 1
        while j + 7 <= m
            for k in 0:7
                @inbounds acc += A[i, j+k] * x[j+k]
            end
            j += 8
        end
        for k in j:m
            @inbounds acc += A[i, k] * x[k]
        end
        @inbounds y[i] = acc
    end
end
const _MUL_KERNEL = matmul_vec8!(oneAPIBackend())

# ── GPU reduction kernel (avoids oneMKL sum which is broken) ──
@kernel function reduce_sum_kernel!(out, x)
    i = @index(Global, Linear)
    if i == 1
        s = zero(eltype(x))
        for j in 1:length(x)
            s += x[j]
        end
        out[1] = s
    end
end
const _SUM_KERNEL = reduce_sum_kernel!(oneAPIBackend())

gpu_mul!(y, A, x) = begin
    n, m = size(A)
    _MUL_KERNEL(y, A, x, n, m; ndrange=(n,))
    oneAPI.oneL0.synchronize()
    return y
end

gpu_sum(x) = begin
    buf = oneArray{eltype(x)}(undef, 1)
    _SUM_KERNEL(buf, x; ndrange=(1,))
    oneAPI.oneL0.synchronize()
    return Array(buf)[1]
end

# ── Load model via Inferno ──
@info "Loading Inferno..."
using Inferno

using Inferno.ModelCPU
using Inferno.Tokenizer: BPETokenizer, encode, decode
using Inferno.LoaderCPU: extract_tensor_cpu

function main()
    gguf = ARGS[1]
    
    devs = oneAPI.devices()
    oneAPI.device!(devs[1])
    @info "Device: $(devs[1])"
    
    # Load model (CPU, Float32)
    @info "Loading model..."
    cpu_model, tok = Inferno.load_model_cpu(gguf)
    cfg = cpu_model.config
    
    h = cfg.hidden_size
    d_ff = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers
    @info "Config: $n_layers layers, $h hidden, $(cfg.vocab_size) vocab"
    
    # Run CPU forward for reference (first token only)
    prompt = "What is 2 + 2 ?"
    tokens = encode(tok, prompt)
    @info "Prompt: \"$prompt\" → $(length(tokens)) tokens: $(tokens)"
    
    @info "Running CPU reference forward (1st token only)..."
    caches = [ModelCPU.init_kv_cache_cpu(cfg, 64) for _ in 1:n_layers]
    cpu_logits = ModelCPU.forward_cpu!(cpu_model, [tokens[1]], 1, caches)
    cpu_logits = cpu_logits[:]
    @info "CPU forward ✓ (top token: $(argmax(cpu_logits)-1))"
    
    # Now test GPU on just ONE layer (first SSM layer + MLP)
    @info "Testing GPU forward on embedding + first layer..."
    
    to_gpu(x) = x isa AbstractMatrix ? oneArray{Float32}(x) : oneArray{Float32}(vec(x))
    
    @info "Uploading weights..."
    embed_gpu = to_gpu(cpu_model.embed)
    layer = cpu_model.layers[1]
    
    # Upload MLP weights
    mlp = layer.mlp
    mlp_gate = to_gpu(mlp.gate_weight)
    mlp_up = to_gpu(mlp.up_weight)
    mlp_down = to_gpu(mlp.down_weight)
    
    # Upload norm weights
    in_norm_w = to_gpu(layer.in_norm.weight)
    
    # Upload final_norm & lm_head
    final_norm_w = to_gpu(cpu_model.final_norm.weight)
    lm_head_gpu = to_gpu(cpu_model.lm_head)
    
    # GPU buffers
    buf_h = oneArray{Float32}(undef, h)
    buf_norm = oneArray{Float32}(undef, h)
    buf_gate = oneArray{Float32}(undef, d_ff)
    buf_up = oneArray{Float32}(undef, d_ff)
    buf_out = oneArray{Float32}(undef, h)
    
    # Warmup: JIT-compile kernels
    @info "Warming up GPU..."
    tiny_a = oneArray{Float16}(rand(Float32, 4, 4))
    tiny_x = oneArray{Float16}(rand(Float32, 4))
    gpu_mul!(oneArray{Float16}(undef, 4), tiny_a, tiny_x)
    gpu_sum(oneArray{Float32}(rand(Float32, h)))
    @info "Warmup done"
    
    token = tokens[1]
    
    # Time individual operations
    @info "Timing individual operations..."
    
    # GPU forward: embedding + first layer
    t0 = time()
    
    # Embed
    copyto!(buf_h, view(embed_gpu, :, token))
    @info "  Embed: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # In-norm
    sum_sq = gpu_sum(buf_h .^ 2)
    @info "  Sum_sq: $(round((time()-t0)*1000))ms"
    t0 = time()
    inv_rms = 1.0f0 / sqrt(sum_sq / h + cfg.rms_norm_eps)
    buf_norm .= buf_h .* inv_rms .* in_norm_w
    @info "  Scale: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # MLP gate projection
    gpu_mul!(buf_gate, mlp_gate, buf_norm)
    @info "  Gate matmul: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # Verify gate matmul against CPU
    cpu_gate_ref = mlp.gate_weight * Array(buf_norm)
    gpu_gate = Array(buf_gate)
    gate_sim = dot(gpu_gate, cpu_gate_ref) / (norm(gpu_gate) * norm(cpu_gate_ref))
    @info "  Gate matmul cosine sim: $(round(gate_sim, digits=6))"
    @info "  GPU gate range: $(round(minimum(gpu_gate), digits=4))..$(round(maximum(gpu_gate), digits=4))"
    @info "  CPU gate range: $(round(minimum(cpu_gate_ref), digits=4))..$(round(maximum(cpu_gate_ref), digits=4))"
    
    gpu_mul!(buf_up, mlp_up, buf_norm)
    @info "  Up matmul: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # SiLU + multiply
    buf_gate .= buf_gate ./ (1.0f0 .+ exp.(-buf_gate))
    buf_gate .= buf_gate .* buf_up
    @info "  SiLU+Mul: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # Down projection
    gpu_mul!(buf_out, mlp_down, buf_gate)
    @info "  Down matmul: $(round((time()-t0)*1000))ms"
    t0 = time()
    
    # Residual
    buf_h .= buf_h .+ buf_out
    @info "  Residual: $(round((time()-t0)*1000))ms"
    
    t_gpu = time() - t0
    @info "GPU 1-layer forward: $(round(t_gpu*1000, digits=1)) ms"
    
    # Compute what the hidden state should be after layer 1
    cpu_hidden_ref = copy(cpu_model.embed_buf)
    # The embed buffer was set during forward_cpu! - get first token embedding
    copy!(cpu_hidden_ref, view(cpu_model.embed, :, tokens[1]))
    cpu_model.layers[1](cpu_hidden_ref, 1, cpu_model.rope, caches[1])
    
    gpu_hidden = Array(buf_h)
    
    cos_sim = dot(gpu_hidden, cpu_hidden_ref) / (norm(gpu_hidden) * norm(cpu_hidden_ref))
    @printf("Hidden state similarity: %.6f\n", cos_sim)
    
    if cos_sim > 0.9
        println("\n✓ GPU layer-1 matches CPU! MLP+Norm pipeline correct.")
        @info "Full GPU model is feasible — need optimized lm_head matmul."
    else
        println("\n✗ GPU diverges — investigating...")
        @printf("Top-10 GPU hidden:\n")
        for i in sortperm(gpu_hidden, rev=true)[1:10]
            @printf("  [%5d] GPU=%.4f  CPU=%.4f\n", i, gpu_hidden[i], cpu_hidden_ref[i])
        end
    end
end

main()