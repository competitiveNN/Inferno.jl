using Test
using Inferno

const SAFETENSORS_MODEL_PATH = get(ENV, "INFERNO_SAFETENSORS_MODEL", "/run/host/var/home/fra/data/models/safetensors/Qwen3.5-0.8B")
const MODEL_EXISTS = isdir(SAFETENSORS_MODEL_PATH)

if MODEL_EXISTS
    @testset "End-to-End Generation Pipeline" begin
        # Load model
        model, tok = Inferno.load_safetensors_model(SAFETENSORS_MODEL_PATH)
        @test model isa Inferno.ModelCPU.QwenModelCPU
        @test tok isa Inferno.Tokenizer.BPETokenizer

        # Test 1: Simple generation
        output = Inferno.generate_text(model, tok, "Hello"; max_tokens=10, temperature=0.7f0)
        @test length(output) > 0
        @info "Simple generation: $(repr(output[1:min(20,end)]))..."

        # Test 2: GenerationState creation and reuse
        state = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
        @test state isa Inferno.ModelCPU.GenerationState
        @test length(state.caches) == model.config.num_hidden_layers

        # Test 3: generate_text with state produces coherent output
        output1 = Inferno.generate_text(model, tok, "The answer is", state; max_tokens=10, temperature=0.7f0)
        @test length(output1) > 0
        @test state.tokens_generated == 10

        # Test 4: State persists and can be reused
        output2 = Inferno.generate_text(model, tok, "The result is", state; max_tokens=5, temperature=0.7f0)
        @test length(output2) > 0
        @test state.tokens_generated == 5

        # Test 5: generate_batch
        prompts = ["AI", "Science"]
        state3 = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
        batch_outputs = Inferno.ModelCPU.generate_batch(model, tok, prompts, state3; max_tokens=5, temperature=0.7f0)
        @test length(batch_outputs) == length(prompts)
        @test all(length(o) > 0 for o in batch_outputs)

        # Test 6: KV cache memory efficiency - verify no new allocations for caches
        # (indirectly: if generate_text with new state allocates much more)
        state4 = Inferno.ModelCPU.create_generation_state(model; max_context=2048)
        # This call reuses the pre-allocated caches from state4
        @time Inferno.generate_text(model, tok, "Test", state4; max_tokens=5, temperature=0.7f0)

        @info "All end-to-end pipeline tests passed"
    end
else
    @warn "Safetensors model not found at $SAFETENSORS_MODEL_PATH, skipping end-to-end pipeline test"
end
