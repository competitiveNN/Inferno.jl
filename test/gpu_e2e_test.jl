# Minimal GPU e2e test for Qwen3.5 — uses ONLY KA kernels, no GPU broadcasts
# Run: julia --project=. test/gpu_e2e_test.jl <model.gguf>

using oneAPI
using oneAPI: oneAPIBackend
using KernelAbstractions: @kernel, @index
using LinearAlgebra
using Printf

# ── GPU matmul kernel (vec8 unrolled, proven correct) ──
@kernel function matmul_vec8!(y, A, x, n::Int, m::Int)
    i = @index(Global, Linear)
    if i <= n
        T = eltype(y)
        acc = zero(T)
        j = 1
        while j + 7 <= m
            @inbounds acc += A[i, j+0] * x[j+0] +
                              A[i, j+1] * x[j+1] +
                              A[i, j+2] * x[j+2] +
                              A[i, j+3] * x[j+3] +
                              A[i, j+4] * x[j+4] +
                              A[i, j+5] * x[j+5] +
                              A[i, j+6] * x[j+6] +
                              A[i, j+7] * x[j+7]
            j += 8
        end
        for k in j:m
            @inbounds acc += A[i, k] * x[k]
        end
        @inbounds y[i] = acc
    end
end
const _MUL = matmul_vec8!(oneAPIBackend())

function gpu_mul!(y, A, x)
    n, m = size(A)
    _MUL(y, A, x, n, m; ndrange=(n,))
    oneAPI.oneL0.synchronize()
end

# ── Reduction: RMSNorm sum-of-squares ──
@kernel function sum_sq_kernel!(out, x)
    i = @index(Global, Linear)
    if i == 1
        s = zero(eltype(x))
        for j in 1:length(x)
            s += x[j] * x[j]
        end
        out[1] = s
    end
end
const _SUM_SQ = sum_sq_kernel!(oneAPIBackend())

function gpu_rmsnorm_sum_sq(x)
    out = oneArray{Float32}(undef, 1)
    _SUM_SQ(out, x; ndrange=(1,))
    oneAPI.oneL0.synchronize()
    return Array(out)[1]
end

# ── Scale: buf[i] = x[i] * scale * w[i] (replaces broadcast norm) ──
@kernel function scale_kernel!(out, x, scale, w)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = x[i] * scale * w[i]
    end
end
const _SCALE = scale_kernel!(oneAPIBackend())

function gpu_scale!(out, x, scale, w)
    _SCALE(out, x, scale, w; ndrange=(length(out),))
    oneAPI.oneL0.synchronize()
end

# ── SiLU: out[i] = out[i] / (1 + exp(-out[i])) in-place ──
@kernel function silu_kernel!(out)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = out[i] / (1.0f0 + exp(-out[i]))
    end
end
const _SILU = silu_kernel!(oneAPIBackend())

function gpu_silu!(x)
    _SILU(x; ndrange=(length(x),))
    oneAPI.oneL0.synchronize()
end

# ── Element-wise mul: out[i] = a[i] * b[i] ──
@kernel function elem_mul_kernel!(out, a, b)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = a[i] * b[i]
    end
end
const _ELEM_MUL = elem_mul_kernel!(oneAPIBackend())

function gpu_elem_mul!(out, a, b)
    _ELEM_MUL(out, a, b; ndrange=(length(out),))
    oneAPI.oneL0.synchronize()
end

# ── Residual add: x[i] = x[i] + d[i] ──
@kernel function residual_add_kernel!(x, d)
    i = @index(Global, Linear)
    if i <= length(x)
        @inbounds x[i] = x[i] + d[i]
    end
end
const _RES_ADD = residual_add_kernel!(oneAPIBackend())

function gpu_residual_add!(x, d)
    _RES_ADD(x, d; ndrange=(length(x),))
    oneAPI.oneL0.synchronize()
end

# ── Copy: out[i] = x[i] ──
@kernel function gpu_copy_kernel!(out, x)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = x[i]
    end
end
const _COPY = gpu_copy_kernel!(oneAPIBackend())

function gpu_copy!(out, x)
    _COPY(out, x; ndrange=(length(out),))
    oneAPI.oneL0.synchronize()
end

# ── Load model ──
@info "Loading Inferno..."
using Inferno
using Inferno.ModelCPU
using Inferno.Tokenizer: BPETokenizer, encode, decode

function main()
    gguf = ARGS[1]

    devs = oneAPI.devices()
    oneAPI.device!(devs[1])
    @info "Device: $(devs[1])"

    @info "Loading model..."
    cpu_model, tok = Inferno.load_model_cpu(gguf)
    cfg = cpu_model.config
    h = cfg.hidden_size
    d_ff = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers
    @info "Config: $n_layers layers, $h hidden, $(cfg.vocab_size) vocab"

    prompt = "What is 2 + 2 ?"
    tokens = encode(tok, prompt)
    @info "Prompt: \"$prompt\" → $(length(tokens)) tokens: $(tokens)"

    # CPU reference: first token only
    @info "Running CPU reference (1st token)..."
    caches = [ModelCPU.init_kv_cache_cpu(cfg, 64) for _ in 1:n_layers]
    cpu_logits = ModelCPU.forward_cpu!(cpu_model, [tokens[1]], 1, caches)
    @info "CPU forward ✓"

    # Compute CPU MLP-only reference for comparison
    # (GPU test currently does in_norm → MLP → residual, skipping SSM)
    @info "Computing CPU MLP-only reference..."
    layer = cpu_model.layers[1]
    mlp = layer.mlp
    cpu_embed = copy(view(cpu_model.embed, :, tokens[1]))
    cpu_hidden_mlp = copy(cpu_embed)
    ModelCPU.rmsnorm_cpu!(cpu_hidden_mlp, cpu_hidden_mlp, layer.in_norm)
    mlp_buf = Vector{Float32}(undef, d_ff)
    mlp_out = Vector{Float32}(undef, h)
    mul!(mlp_buf, mlp.gate_weight, cpu_hidden_mlp)
    s = mlp_buf ./ (1.0f0 .+ exp.(-mlp_buf))
    mul!(mlp_buf, mlp.up_weight, cpu_hidden_mlp)
    mlp_buf .= s .* mlp_buf
    mul!(mlp_out, mlp.down_weight, mlp_buf)
    cpu_hidden_mlp = cpu_embed .+ mlp_out
    @info "  Layer 1: $(typeof(layer.op).name.name)"

    # ── Upload weights to GPU ──
    @info "Uploading weights..."
    to_gpu(x) = oneArray{Float32}(x)
    embed_gpu = to_gpu(cpu_model.embed)
    mlp_gate = to_gpu(mlp.gate_weight)
    mlp_up   = to_gpu(mlp.up_weight)
    mlp_down = to_gpu(mlp.down_weight)
    in_norm_w = to_gpu(layer.in_norm.weight)

    # GPU buffers
    buf_h = oneArray{Float32}(undef, h)
    buf_n = oneArray{Float32}(undef, h)
    buf_g = oneArray{Float32}(undef, d_ff)
    buf_u = oneArray{Float32}(undef, d_ff)

    # Warmup JIT
    @info "Warming up JIT..."
    tiny = oneArray{Float32}(rand(Float32, 4, 4))
    tiny2 = oneArray{Float32}(rand(Float32, 4))
    tiny3 = oneArray{Float32}(undef, 4)
    gpu_mul!(tiny3, tiny, tiny2)
    gpu_rmsnorm_sum_sq(tiny3)
    gpu_scale!(tiny3, tiny3, 0.5f0, tiny3)
    gpu_silu!(tiny3)
    gpu_elem_mul!(tiny3, tiny3, tiny3)
    gpu_residual_add!(tiny3, tiny2)
    gpu_copy!(tiny3, tiny2)
    @info "JIT warmup done."

    # ════════════════════════════════════════════════════
    # GPU forward: embedding + layer 1 (no broadcasts!)
    # ════════════════════════════════════════════════════
    t0 = time()

    # Embed via copy kernel
    gpu_copy!(buf_h, view(embed_gpu, :, tokens[1]))
    @info "  Embed: $(round((time()-t0)*1000))ms"
    t0 = time()

    # RMSNorm (in-place on buf_n)
    sum_sq = gpu_rmsnorm_sum_sq(buf_h)
    inv_rms = 1.0f0 / sqrt(sum_sq / h + cfg.rms_norm_eps)
    gpu_scale!(buf_n, buf_h, inv_rms, in_norm_w)
    @info "  RMSNorm: $(round((time()-t0)*1000))ms"
    t0 = time()

    # MLP gate
    gpu_mul!(buf_g, mlp_gate, buf_n)
    @info "  Gate matmul: $(round((time()-t0)*1000))ms"
    t0 = time()

    # Verify matmul against CPU
    cpu_g = mlp.gate_weight * Array(buf_n)
    gpu_g = Array(buf_g)
    sim = dot(gpu_g, cpu_g) / (norm(gpu_g) * norm(cpu_g))
    @info "  Gate cosim: $(round(sim, digits=6))"

    # MLP up
    gpu_mul!(buf_u, mlp_up, buf_n)
    @info "  Up matmul: $(round((time()-t0)*1000))ms"
    t0 = time()

    # SiLU on gate (in-place) — KA kernel, no broadcast!
    gpu_silu!(buf_g)
    @info "  SiLU (KA kernel): $(round((time()-t0)*1000))ms"
    t0 = time()

    # Element-wise: gate .* up — KA kernel, no broadcast!
    gpu_elem_mul!(buf_g, buf_g, buf_u)
    @info "  Gate*Up (KA kernel): $(round((time()-t0)*1000))ms"
    t0 = time()

    # Down projection
    gpu_mul!(buf_n, mlp_down, buf_g)
    @info "  Down matmul: $(round((time()-t0)*1000))ms"
    t0 = time()

    # Residual add — KA kernel, no broadcast!
    gpu_residual_add!(buf_h, buf_n)
    @info "  Residual add: $(round((time()-t0)*1000))ms"

    total = (time() - t0)
    @info "GPU 1-layer forward: $(round(total*1000, digits=1))ms"

    gpu_hidden = Array(buf_h)
    cos_sim = dot(gpu_hidden, cpu_hidden_mlp) /
              (norm(gpu_hidden) * norm(cpu_hidden_mlp))
    @printf("\nHidden state similarity (vs CPU MLP-only): %.6f\n", cos_sim)

    if cos_sim > 0.99
        println("✓ GPU matches CPU! GPU pipeline is correct!")
    elseif cos_sim > 0.9
        println("~ GPU close (cosim=$(round(cos_sim, digits=4)))")
        @printf("Max diff: %.6f\n", maximum(abs.(gpu_hidden .- cpu_hidden_mlp)))
    else
        println("✗ GPU diverges from MLP-only forward")
        @printf("Top-10 GPU hidden vs CPU (MLP-only):\n")
        for i in sortperm(gpu_hidden, rev=true)[1:10]
            @printf("  [%5d] GPU=%.6f  CPU=%.6f\n", i, gpu_hidden[i], cpu_hidden_mlp[i])
        end
    end
end

main()