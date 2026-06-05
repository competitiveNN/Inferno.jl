# Open Pull Requests Summary

Kept 4 most recent PRs with pending improvements:

## PR #85: CPU Inference Optimizations ✅ MERGED
- **FlashAttention**: Pre-allocated scores buffer, @turbo for vectorization, fused block max
- **lm_head_project**: Zero-allocation with `@sync`/`Threads.@spawn`, direct slice writes
- **sampling**: `partialsortperm` instead of full sort for top_k filtering

## PR #86: lm_head Optimization
- Eliminates allocations in `lm_head_project!`
- Chunked parallel execution with direct output buffer writes

## PR #88: Palette UX Improvements (Chat CLI)
- Interactive chat UX fixes
- Bug fixes for streaming

## PR #89: softmax_sample Filtering
- Partial sort instead of full sort (O(N log k) vs O(N log N))
- Eliminates Set allocations in hot sampling path

---

Closed 26 stale PRs:
- PRs #60-84, 87, 90+: Older iterations on same themes (superseded by newer)
- PRs #24-59: GPU optimization PRs (CPU-only backend is focus)
- PRs #30-33: Loading feedback (now handled in #88)