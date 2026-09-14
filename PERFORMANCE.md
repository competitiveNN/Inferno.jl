# Inferno.jl Performance Baseline

## Environment

- **Julia**: 1.13.0
- **CPU**: Intel Arrow Lake / hybrid, 20 cores
- **BLAS threads**: 8
- **Model**: Qwen3.5-0.8B safetensors (24 layers, hidden=1024, vocab=248320)
- **Backend**: CPU only (oneAPI available but CPU path is working)
- **Date**: 2026-09-14

## Benchmark Prompts

Primary benchmark prompt: `"The capital of France is"` (1-2 tokens)

## Throughput Results

### Streaming Path (`stream_to_stdout_cpu`)

Reuses caches internally via `generate_stream_cpu`. Each call creates its own KV caches.

| Run | Time (s) | Tokens | tok/s |
|-----|----------|--------|-------|
| 1 | 1.576 | 30 | 19.0 |
| 2 | 1.560 | 30 | 19.2 |
| 3 | 1.572 | 30 | 19.1 |
| **Average** | **1.569** | **30** | **19.1** |

### High-Level API (`generate_text`)

Calls `stream_to_stdout_cpu` internally; includes tokenization overhead.

| Run | Time (s) | Tokens | tok/s |
|-----|----------|--------|-------|
| 1 | 1.931 | 30 | 15.5 |
| 2 | ~1.95 | 30 | ~15.4 |
| **Average** | **~1.95** | **30** | **~15.4** |

### Low-Level Manual Loop

Reuses KV caches and forward pass. Fastest path.

| Tokens | Time (s) | tok/s |
|--------|----------|-------|
| 10 | 0.434 | 23.0 |

### Per-Token Latency (Greedy, `generate_text`)

Measures single-token generation including cache initialization:

| Token | Time (ms) |
|-------|-----------|
| 1 | 208.9 |
| 2 | 281.8 |
| 3 | 290.6 |
| 4 | 286.6 |
| 5 | 285.7 |

**Average per-token**: ~270 ms (includes repeated cache initialization overhead)

## Memory Profile

### Per-Token Generation (`generate_text`, 1 tok, greedy)

```
  0.632 seconds (388.13 k allocations: 800.003 MiB, 43% compilation time)
```

This is per single-token generation call. The high allocation count is dominated by:
1. KV cache allocation in `init_kv_cache_cpu` (called per generation)
2. Channel creation in `generate_stream_cpu`
3. Tokenizer encode in `generate_text`
4. Temporary arrays in `forward_cpu!` and `lm_head_project!`

### Per-Token Generation (`generate_stream_cpu`, 10 tokens, streaming)

```
  1.044s for 10 tokens — 9.6 tok/s
```

With warmup (first call excluded), subsequent 10-token generations show:
- ~0.85s per 10 tokens
- ~11.8 tok/s

### Known Hot Spots

1. **KV cache init** (`init_kv_cache_cpu`) — allocates large buffers each generation call. Pre-allocate and reuse across calls for significant speedup.
2. **Tokenizer.encode** — called per `generate_text` call. Not reused across generations.
3. **Channel creation** — `generate_stream_cpu` creates caches inside `Channel` block. Pre-allocate outside.
4. **lm_head matmul** — dominant compute time (~47% of total). Chunked parallel implementation in `lm_head_project!` with 4 chunks.
5. **Attention computation** — flash attention enabled by default, provides 8-13.7x speedup vs standard attention.

## KV Cache Optimization Results

### Persistent KV Cache API

Implemented `GenerationState` struct with pre-allocated KV caches and `create_generation_state()` / `generate_with_cache()` functions.

**Allocation comparison** (generating 10 tokens, Qwen3.5-0.8B, max_context=2048):

| Approach | Time | Allocations | Memory |
|----------|------|-------------|--------|
| `generate_text` (new caches each call) | ~0.7s | ~1.3k | ~325 MB |
| `generate_with_cache` (reuses caches) | ~0.6s | ~1.1k | ~132 MB |

- **60% reduction in memory allocations** (~193 MB saved per call)
- **~15% throughput improvement** (0.6s vs 0.7s per 10 tokens)
- KV caches persist across calls; `reset_state!` resets counters but preserves buffers
- API: `state = create_generation_state(model; max_context=8192)` then `generate_text(model, tok, prompt, state; ...)`

### Optimization Targets

#### Achieved

- Per-token allocation: 2.7MB → 10KB (99.6% reduction) — but only in the optimized forward path
- Pre-allocated buffers for all major operations
- Flash attention integration with 8-13.7x speedup
- BLAS thread tuning: 8 threads optimal
- **Persistent KV cache API** (`GenerationState`, `create_generation_state`, `generate_with_cache`) — 60% memory reduction

### Remaining Opportunities

1. **Tokenizer reuse** — `encode` called per generation. Cache tokenized prompts.
2. **Channel pre-allocation** — Move `generate_stream_cpu` cache creation outside the `Channel` block.
3. **Batch generation** — Support generating multiple sequences in a single forward pass.

## Profiling Commands

```bash
# Quick throughput check (30 tokens, sampling)
julia --project=. /tmp/perf.jl

# Memory allocation breakdown
julia --project=. /tmp/bench4.jl

# Full test suite
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Verified Output Quality

Generation produces coherent multi-token text matching HuggingFace Qwen3.5 reference output patterns. Verified with prompts including:
- `"What is 2 + 2 ?"` → `2 + 2 = 4 ...`
- `"The capital of France is"` → coherent text about French geography
- Minimum 64-128 tokens verified for coherence
