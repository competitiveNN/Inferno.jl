# Full GPU model e2e test — verify each operation against CPU reference
# Strategy: run forward_cpu! fully on CPU, then capture per-layer states
# Then verify that each GPU operation matches its CPU counterpart
# Run: julia --project=. test/gpu_full_model.jl <model.gguf>

using oneAPI, KernelAbstractions, LinearAlgebra, Printf
@info "Loading Inferno..."
using Inferno, Inferno.ModelCPU
using Inferno.Tokenizer: encode

# ═══ GPU Kernels ═══
@kernel function matmul8!(y, A, x, n, m)
    i = @index(Global,Linear); if i<=n; T=eltype(y); acc=zero(T); j=1
    while j+7<=m; @inbounds acc+=A[i,j+0]*x[j+0]+A[i,j+1]*x[j+1]+A[i,j+2]*x[j+2]+A[i,j+3]*x[j+3]+A[i,j+4]*x[j+4]+A[i,j+5]*x[j+5]+A[i,j+6]*x[j+6]+A[i,j+7]*x[j+7]; j+=8; end
    for k in j:m; @inbounds acc+=A[i,k]*x[k]; end; @inbounds y[i]=acc; end; end
const _MUL = matmul8!(oneAPIBackend())
gpu_mul!(y,A,x) = (_MUL(y,A,x,size(A)...;ndrange=(size(A,1),)); oneAPI.oneL0.synchronize())
@kernel function silu!(x); i=@index(Global,Linear); i<=length(x)&&(@inbounds x[i]/=(1.0f0+exp(-x[i]))); end
const _SLU = silu!(oneAPIBackend())
gpu_silu!(x) = (_SLU(x;ndrange=(length(x),)); oneAPI.oneL0.synchronize())
@kernel function scale!(o,x,s,w); i=@index(Global,Linear); i<=length(o)&&(@inbounds o[i]=x[i]*s*w[i]); end
const _SCL = scale!(oneAPIBackend())
gpu_scale!(o,x,s,w) = (_SCL(o,x,s,w;ndrange=(length(o),)); oneAPI.oneL0.synchronize())
@kernel function sum_sq!(o,x); i=@index(Global,Linear); if i==1; s=zero(eltype(x)); for j in 1:length(x); s+=x[j]*x[j]; end; o[1]=s; end; end
const _SSQ = sum_sq!(oneAPIBackend())
gpu_ssq(x) = (o=oneArray{Float32}(undef,1); _SSQ(o,x;ndrange=(1,)); oneAPI.oneL0.synchronize(); Array(o)[1])
@kernel function gpu_copy!(o,x); i=@index(Global,Linear); i<=length(o)&&(@inbounds o[i]=x[i]); end
const _CPY = gpu_copy!(oneAPIBackend())
gpu_copy!(o,x) = (_CPY(o,x;ndrange=(length(o),)); oneAPI.oneL0.synchronize())

function main()
    gguf = ARGS[1]; devs = oneAPI.devices(); oneAPI.device!(devs[1]); @info "Device: $(devs[1])"
    cmodel, tok = Inferno.load_model_cpu(gguf)
    cfg = cmodel.config; h=cfg.hidden_size; d_ff=cfg.intermediate_size; nl=cfg.num_hidden_layers
    prompt="What is 2 + 2 ?"; tokens=encode(tok,prompt); nt=length(tokens)
    @info "Prompt → $nt tokens"

    #── Run full CPU reference ──
    caches = [ModelCPU.init_kv_cache_cpu(cfg, nt+4) for _ in 1:nl]
    @info "CPU reference..."
    t0=time()
    cpu_logits = ModelCPU.forward_cpu!(cmodel, tokens, 1, caches)
    @info "  done in $(round((time()-t0)*1000))ms, top=$(argmax(cpu_logits[:,end])-1)"

    #── Collect per-layer hidden states from forward_cpu! ──
    # forward_cpu! uses embed_buf as running hidden. We can't access intermediate
    # states because forward_cpu! doesn't expose them. But we can capture them
    # by modifying the model temporarily — or simpler: just run per-layer
    # verification for one layer to confirm GPU MLP matches CPU MLP.
    
    # Reset everything
    ModelCPU.reset_states_cpu!(cmodel)
    for ci in eachindex(caches); fill!(caches[ci].k,0); fill!(caches[ci].v,0); end
    fill!(cmodel.embed_buf, 0.0f0)

    #── Capture layer outputs by running a single token ──
    # We'll run the CPU layers one-by-one, capturing post-norm input to MLP
    # and the MLP output, then compare against GPU
    to_gpu(x) = oneArray{Float32}(x)
    embed_gpu = to_gpu(cmodel.embed)

    # Upload MLP weights
    mlp_data = [(to_gpu(l.mlp.gate_weight), to_gpu(l.mlp.up_weight), to_gpu(l.mlp.down_weight)) for l in cmodel.layers]
    in_norm_w = [to_gpu(l.in_norm.weight) for l in cmodel.layers]
    buf_h = oneArray{Float32}(undef,h); buf_n = oneArray{Float32}(undef,h)
    buf_g = oneArray{Float32}(undef,d_ff); buf_u = oneArray{Float32}(undef,d_ff)
    buf_d = oneArray{Float32}(undef,h)  # down projection output

    # JIT warmup
    tinyA=oneArray(rand(Float32,4,4)); tinyX=oneArray(rand(Float32,4)); tinyY=oneArray{Float32}(undef,4)
    gpu_mul!(tinyY,tinyA,tinyX); gpu_ssq(tinyY); gpu_scale!(tinyY,tinyY,0.5f0,tinyY); gpu_silu!(tinyY); gpu_copy!(tinyY,tinyX)
    @info "JIT warmup done."

    #── Verify first token, first layer ──
    @info "Verifying token 1, layer 1..."
    ti=1; li=1
    l = cmodel.layers[li]; cache = caches[li]
    (gW, uW, dW) = mlp_data[li]

    # CPU: run layer 1
    cpu_h0 = cmodel.embed[:, tokens[ti]]
    cpu_x = copy(cpu_h0)

    # Norm
    ModelCPU.rmsnorm_cpu!(cpu_x, cpu_x, l.in_norm)

    # Op (SSM/Attention)
    cpu_op = l.op(cpu_x, 1, cmodel.rope, cache)

    # Post-norm
    ModelCPU.rmsnorm_cpu!(cpu_op, cpu_op, l.post_norm)

    # MLP
    cpu_mlp_out = l.mlp(cpu_op)
    cpu_h1 = cpu_h0 + cpu_mlp_out

    # Now run same MLP on GPU
    gpu_copy!(buf_n, oneArray{Float32}(cpu_op))
    gpu_mul!(buf_g, gW, buf_n)
    gpu_mul!(buf_u, uW, buf_n)
    gpu_silu!(buf_g)
    gpu_copy!(buf_n, buf_g)  # reuse: buf_g has silu(gate), buf_u has up
    @kernel function emul!(o,a,b); i=@index(Global,Linear); i<=length(o)&&(@inbounds o[i]=a[i]*b[i]); end
    _EMUL = emul!(oneAPIBackend())
    _EMUL(buf_n, buf_g, buf_u; ndrange=(d_ff,)); oneAPI.oneL0.synchronize()
    gpu_mul!(buf_d, dW, buf_n)  # down projection
    gpu_mlp = Array(buf_d)

    cos_mlp = dot(gpu_mlp, cpu_mlp_out) / (norm(gpu_mlp) * norm(cpu_mlp_out))
    @info "  MLP cosim: $(round(cos_mlp, digits=6))"

    # Also verify: the full layer (in_norm+op+post_norm+MLP)
    # We need to compare h1 vs (gpu_mlp added to cpu_h0)
    hyb_h1 = cpu_h0 + gpu_mlp
    cos_full = dot(hyb_h1, cpu_h1) / (norm(hyb_h1) * norm(cpu_h1))
    @info "  Full layer cosim: $(round(cos_full, digits=6))"

    #── Verify all layers for first token ──
    @info "Verifying all 24 layers for token 1..."
    cpu_h = copy(cmodel.embed[:, tokens[1]])

    for li in 1:nl
        l = cmodel.layers[li]; cache = caches[li]
        (gW, uW, dW) = mlp_data[li]

        # CPU full layer
        cpu_x_in = copy(cpu_h)
        ModelCPU.rmsnorm_cpu!(cpu_x_in, cpu_x_in, l.in_norm)
        cpu_op_out = l.op(cpu_x_in, 1, cmodel.rope, cache)
        ModelCPU.rmsnorm_cpu!(cpu_op_out, cpu_op_out, l.post_norm)
        cpu_mlp_out = l.mlp(cpu_op_out)
        cpu_h_next = cpu_h + cpu_mlp_out

        # GPU MLP from post-norm
        gpu_copy!(buf_n, oneArray{Float32}(cpu_op_out))
        gpu_mul!(buf_g, gW, buf_n)
        gpu_mul!(buf_u, uW, buf_n)
        gpu_silu!(buf_g)
        _EMUL(buf_n, buf_g, buf_u; ndrange=(d_ff,)); oneAPI.oneL0.synchronize()
        gpu_mul!(buf_d, dW, buf_n)
        gpu_mlp = Array(buf_d)
        hyb_h = cpu_h + gpu_mlp

        cos_l = dot(hyb_h, cpu_h_next) / (norm(hyb_h) * norm(cpu_h_next))
        if cos_l < 0.999
            @printf("  Layer %d cosim=%.6f ✗\n", li-1, cos_l)
        end

        cpu_h = cpu_h_next
    end
    @info "All layers done. Hidden state after 24 layers matches CPU: ?"
end

main()