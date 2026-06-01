push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Inferno
using Inferno.Model
using Inferno.QuantsData
using oneAPI
using BenchmarkTools

function benchmark()
    config = QwenConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=2, # Reduced for benchmark
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
    )

    println("Initializing mock model...")
    # Mock data for IQ2XXS
    data = rand(UInt8, 66 * (1024 * 1024 ÷ 256))

    # Initialize GPU tables
    Model.init_gpu_tables(QuantsData.IQ2XXS_GRID, QuantsData.KSIGNS_IQ2XS, QuantsData.KMASK_IQ2XS)

    # Create a mock layer
    function create_mock_layer(config)
        # Use random matrices, but some are IQ2XXSMatrix
        # For simplicity, let's use Float32 for all but one to see the impact
        # Actually, let's use what the real model uses
        qw = IQ2XXSMatrix(data, 1024, 1024 * 2 * 8) # wq
        kw = IQ2XXSMatrix(data, 1024, 1024 * 2) # wk
        vw = IQ2XXSMatrix(data, 1024, 1024 * 2) # wv
        ow = oneArray(randn(Float32, 1024, 1024)) # wo

        q_norm = RMSNorm(oneArray(ones(Float32, 256)), 1e-6)
        k_norm = RMSNorm(oneArray(ones(Float32, 256)), 1e-6)

        attn = FullAttention(qw, kw, vw, ow, q_norm, k_norm, 8, 2, 256)

        in_norm = RMSNorm(oneArray(ones(Float32, 1024)), 1e-6)
        post_norm = RMSNorm(oneArray(ones(Float32, 1024)), 1e-6)

        gate_w = IQ2XXSMatrix(data, 1024, 3584)
        up_w = IQ2XXSMatrix(data, 1024, 3584)
        down_w = IQ2XXSMatrix(data, 3584, 1024)
        mlp = MLP(gate_w, up_w, down_w)

        return DecoderLayer(in_norm, attn, post_norm, mlp, false)
    end

    layers = [create_mock_layer(config) for _ in 1:config.num_hidden_layers]

    embed = randn(Float32, 1024, config.vocab_size)
    final_norm = RMSNorm(oneArray(ones(Float32, 1024)), 1e-6)
    lm_head = randn(Float32, 1024, config.vocab_size)
    rope = RotaryEmbedding(256)

    model = QwenModel(config, embed, layers, final_norm, lm_head, rope)

    tokens = [1, 2, 3, 4]
    caches = [init_kv_cache(config) for _ in 1:config.num_hidden_layers]

    println("Running benchmark...")
    # Warmup
    forward!(model, tokens, 0, caches)

    # Timing
    t = @belapsed forward!($model, $tokens, 0, $caches)
    println("Average forward pass time (prefill, 4 tokens): ", t * 1000, " ms")

    # Decode step
    t_decode = @belapsed forward!($model, [1], 4, $caches)
    println("Average forward pass time (decode, 1 token): ", t_decode * 1000, " ms")
end

# Check if oneAPI is available
try
    if isempty(collect(oneAPI.devices()))
        println("No oneAPI devices found. Skipping benchmark.")
    else
        benchmark()
    end
catch e
    println("Error during benchmark: ", e)
end
