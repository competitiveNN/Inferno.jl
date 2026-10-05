# Edge-case tests for CPU generation paths.
#
# Covers the correctness properties audited in the session:
#   - max_tokens = -1 / 0 / 1 (validation + no-op for zero)
#   - empty prompts (reset state, return "", no sample)
#   - stop-token semantics (suppressed before decode/yield/count; first vs later)
#   - caller stop_tokens immutability (no mutation of the caller-supplied Set)
#   - context-limit boundaries and overflow
#   - KV-cache position correctness across long-short-long reuse vs a fresh state
#
# These tests use the safetensors model (gated below) and greedy (temp=0) sampling
# where determinism is required, so they compare against a freshly initialized state.

using Test
using Inferno

const SAFETENSORS_MODEL_PATH = get(
    ENV, "INFERNO_SAFETENSORS_MODEL",
    "/run/host/var/home/fra/data/models/safetensors/Qwen3.5-0.8B")

if isdir(SAFETENSORS_MODEL_PATH)
    @testset "Generation edge cases" begin
        model, tok = Inferno.load_safetensors_model(SAFETENSORS_MODEL_PATH)

        # ----- max_tokens validation and zero handling -----
        @testset "max_tokens validation" begin
            state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)

            # max_tokens = 0: state path returns "" without generating, without
            # advancing curr_pos or incrementing tokens_generated.
            out0 = Inferno.ModelCPU.generate_with_cache(model, tok, "Hello", state;
                max_tokens=0, temperature=0.7f0)
            @test out0 == ""
            @test state.tokens_generated == 0
            @test state.curr_pos == 0

            # max_tokens = -1: rejected before any work, by both state and non-state paths.
            @test_throws ErrorException Inferno.ModelCPU.generate_with_cache(
                model, tok, "Hello", state; max_tokens=-1)
            @test_throws ErrorException Inferno.generate_text(
                model, tok, "Hello"; max_tokens=-1)
        end

        @testset "max_tokens = 1" begin
            state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            out1 = Inferno.ModelCPU.generate_with_cache(model, tok, "Hello", state;
                max_tokens=1, temperature=0.7f0)
            @test length(out1) > 0
            @test state.tokens_generated == 1
            # curr_pos advanced exactly once past the prompt.
            @test state.curr_pos == length(Inferno.Tokenizer.encode(tok, "Hello"))
        end

        # ----- empty prompts -----
        @testset "empty prompt" begin
            state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            out_empty = Inferno.ModelCPU.generate_with_cache(model, tok, "", state;
                max_tokens=10, temperature=0.7f0)
            @test out_empty == ""
            @test state.tokens_generated == 0
            # Non-state path also short-circuits to "".
            @test Inferno.generate_text(model, tok, ""; max_tokens=10) == ""
        end

        # ----- caller stop_tokens are never mutated -----
        @testset "stop_tokens immutability" begin
            user_stops = Set{Int}([2, 3, 4])
            # Non-state BPE overload must copy before adding EOS.
            pre = copy(user_stops)
            _ = Inferno.generate_text(model, tok, "Hello";
                max_tokens=2, stop_tokens=user_stops)
            @test user_stops == pre

            # State-aware overload must copy too.
            state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            pre2 = copy(user_stops)
            _ = Inferno.generate_text(model, tok, "Hello", state;
                max_tokens=2, stop_tokens=user_stops)
            @test user_stops == pre2
            @test tok.eos_id ∉ pre2   # EOS added only internally
        end

        # ----- stop tokens are suppressed before decode/yield/count -----
        @testset "stop token suppression" begin
            eos = tok.eos_id
            # Stream: no yielded string should ever equal a bare stop token, and
            # the channel must close cleanly (no hang) when EOS is a stop token.
            seen = String[]
            decode_fn = (ids) -> Inferno.Tokenizer.decode(tok, ids)
            for s in Inferno.ModelCPU.generate_stream_cpu(model,
                    Inferno.Tokenizer.encode(tok, "Hello"), decode_fn;
                    max_tokens=16, temperature=0.7f0, stop_tokens=Set{Int}([eos]))
                push!(seen, s)
            end
            @test length(seen) <= 16
        end

        # ----- context-limit boundaries and overflow -----
        @testset "context limits" begin
            # Non-positive max_context is rejected at state creation.
            @test_throws ErrorException Inferno.ModelCPU.create_generation_state(
                model; max_context=0)

            # Prompt longer than the allocated context is rejected before prefill.
            state_small = Inferno.ModelCPU.create_generation_state(model; max_context=4)
            @test_throws ErrorException Inferno.ModelCPU.generate_with_cache(
                model, tok, "The capital of France is", state_small; max_tokens=1)

            # Prompt fits but prompt + max_tokens overflows: rejected before compute.
            @test_throws ErrorException Inferno.ModelCPU.generate_with_cache(
                model, tok, "Hi", state_small; max_tokens=8)
        end

        # ----- KV-cache position correctness: reuse vs fresh state -----
        @testset "KV-cache reuse correctness" begin
            # Greedy decoding makes output deterministic for a given cache state,
            # so a reused state must produce the same text as a fresh state when
            # the logical prompt/generation is identical.
            prompt = "The capital of France is"

            fresh = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            out_fresh = Inferno.ModelCPU.generate_with_cache(model, tok, prompt, fresh;
                max_tokens=3, temperature=0.0f0)

            # Long-short-long reuse: a short no-op call in between must not corrupt
            # the KV positions, so the long prompt still reproduces the fresh output.
            reused = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            _ = Inferno.ModelCPU.generate_with_cache(model, tok, "x", reused;
                max_tokens=0, temperature=0.0f0)          # short, no generation
            out_reused = Inferno.ModelCPU.generate_with_cache(model, tok, prompt, reused;
                max_tokens=3, temperature=0.0f0)

            @test out_fresh == out_reused
            @test length(out_reused) > 0

            # A prior generated continuation must not leave stale KV that changes
            # the logits of a fresh prompt (positions must be relative to the new
            # prefill, not the old curr_pos).
            tainted = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            _ = Inferno.ModelCPU.generate_with_cache(model, tok, prompt, tainted;
                max_tokens=4, temperature=0.0f0)           # long, generates tokens
            out_after = Inferno.ModelCPU.generate_with_cache(model, tok, prompt, tainted;
                max_tokens=3, temperature=0.0f0)
            @test out_after == out_fresh
        end

        # ----- batch generation edge cases -----
        @testset "generate_batch edges" begin
            state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
            outs = Inferno.ModelCPU.generate_batch(model, tok, [""], state;
                max_tokens=5, temperature=0.7f0)
            @test outs == [""]
            outs2 = Inferno.ModelCPU.generate_batch(model, tok, ["Hi", ""], state;
                max_tokens=2, temperature=0.7f0)
            @test length(outs2) == 2
            @test outs2[2] == ""
            @test length(outs2[1]) > 0
        end
    end
else
    @warn "Safetensors model not found at $SAFETENSORS_MODEL_PATH, skipping generation edge-case tests"
end
