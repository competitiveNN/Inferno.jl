# Inferno.jl

**Status: PAUSED INDEFINITELY.** Development is currently halted due to an unresolved error with the oneAPI/LevelZero GPU stack that cannot be diagnosed from this tree. See the AGENTS.md doc for the GPU backend status.

Julia native inference engine for GGUF models with Intel GPU (oneAPI) support.

## Features

- Pure Julia implementation (no Python dependencies)
- CPU and Intel GPU (oneAPI) backends
- GGUF format support with quantization (Q4_K, Q5_K, Q6_K, Q8_0, etc.)
- Qwen3.5 architecture support (SSM + Full Attention hybrid)

## Quick Start

```julia
using Inferno

# Load model on CPU
model, file = LoaderCPU.load_model_cpu("path/to/model.gguf")

# Initialize KV cache and reset SSM states
caches = [ModelCPU.init_kv_cache_cpu(model.config, 512) for _ in 1:model.config.num_hidden_layers]
ModelCPU.reset_states_cpu!(model)

# Run inference for a single token
x = view(model.embed, :, token_id)  # Get embedding
for (i, layer) in enumerate(model.layers)
    x = layer(x, position, model.rope, caches[i])
end
x = model.final_norm(x)
logits = model.lm_head * x

# Get next token (greedy)
next_token = argmax(logits)
```

## Example

See `examples/basic_usage.jl` for a complete generation example:

```bash
julia --project=. examples/basic_usage.jl
```

## Testing

Run tests with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Model Download

Download a test model:

```bash
./scripts/download_model.sh
```

Or manually download a Qwen3.5 GGUF model from HuggingFace and set the path:

```bash
export INFERNO_MODEL_PATH=/path/to/model.gguf
```

## Architecture Support

### Qwen3.5 (Hybrid SSM + Attention)

- SSM layers (Mamba-like state space model) every 4th layer
- Full attention layers with:
  - Query/Gate gating mechanism
  - Q/K normalization
  - Grouped Query Attention (GQA)
  - Partial rotary embeddings (25% of head_dim)

### Weight Format

The model expects GGUF format with:

- `blk.N.attn_qkv.weight` for SSM layers
- `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight` for attention layers
- `blk.N.attn_output.weight` for output projection
- `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` for normalization

## Development Status

- [x] GGUF loading and parsing
- [x] Dequantization (Q4_K, Q5_K, Q6_K, Q8_0)
- [x] RMSNorm (Qwen3.5 style: 1 + weight)
- [x] Rotary Position Embeddings (partial rotary)
- [x] SSM/Mamba layer
- [x] Full Attention layer with Q/K norm and gating
- [ ] GPU kernel optimization — **paused indefinitely** (unresolved oneAPI/LevelZero error; see note above)
- [ ] Proper tokenizer (BPE)
- [ ] Sampling strategies (temperature, top-p, top-k)
- [ ] Batched inference

## Project Structure

```
src/
├── Inferno.jl         # Main module
├── GGUF.jl            # GGUF file format parsing
├── Dequant.jl         # Quantization dequantization
├── ModelCPU.jl        # CPU inference kernels
├── LoaderCPU.jl       # CPU model loading
├── ModelGPU.jl        # GPU inference kernels (WIP)
└── LoaderGPU.jl       # GPU model loading (WIP)
test/
└── runtests.jl        # Test suite
examples/
└── basic_usage.jl     # Usage example
```

## License

MIT
