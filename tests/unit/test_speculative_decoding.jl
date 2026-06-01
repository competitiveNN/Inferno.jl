# Unit Tests for Speculative Decoding Module
#
# Tests for:
# - sample_from_probs function
# - SpeculativeDecoder type accessibility

using Test
using Random
using Inferno.ModelCPU  # Import the module where SpeculativeDecoder and sample_from_probs are defined

@testset "Speculative Decoding Tests" begin

    @testset "sample_from_probs function" begin
        # Test deterministic case - should always pick the first 1.0
        probs = Float32[0.0, 0.0, 1.0, 0.0]  # Should always pick index 3
        Random.seed!(42)
        result = ModelCPU.sample_from_probs(probs)
        @test result == 3
        
        # Test another deterministic case
        probs = Float32[1.0, 0.0, 0.0, 0.0]  # Should always pick index 1
        Random.seed!(42)
        result = ModelCPU.sample_from_probs(probs)
        @test result == 1
        
        # Test uniform distribution - should get varied results
        probs = Float32[0.25, 0.25, 0.25, 0.25]
        Random.seed!(42)
        results = [ModelCPU.sample_from_probs(probs) for _ in 1:100]
        # Should get a distribution across all indices
        unique_results = unique(results)
        @test length(unique_results) > 1  # Should get multiple different results
        @test all(x -> 1 <= x <= 4, unique_results)  # All should be valid indices
        
        # Test edge case - single element
        probs = Float32[1.0]
        Random.seed!(42)
        result = ModelCPU.sample_from_probs(probs)
        @test result == 1
        
        # Test edge case - all zeros (should return last index)
        probs = Float32[0.0, 0.0, 0.0]
        Random.seed!(42)
        result = ModelCPU.sample_from_probs(probs)
        @test result == 3
    end

    @testset "SpeculativeDecoder type accessibility" begin
        # Test that we can access the SpeculativeDecoder type from ModelCPU
        # This tests that the module is properly loaded and accessible
        @test :SpeculativeDecoder in names(ModelCPU)  # It should be defined in the module
        # Get the type
        SD = ModelCPU.SpeculativeDecoder
        @test typeof(SD) <: Type  # The type of SD should be a Type (since SD is a struct)
        # Check that it has the expected fields (order doesn't matter)
        @test Set(fieldnames(SD)) == Set((:draft_model, :target_model, :gamma))
    end

end