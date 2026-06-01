# Unit Tests for Engine Module
#
# Tests for:
# - sample function (argmax, top-k, top-p)
# - simple_sample
# - mask_and_sample
# - cpu_sample_from_scaled

using Test
using Inferno
using Inferno.Engine
using Random

@testset "Engine Module Tests" begin

    @testset "simple_sample" begin
        # Test basic probability sampling
        Random.seed!(42)
        
        # Deterministic case: single element
        probs1 = Float16[1.0]
        samples1 = [Engine.simple_sample(probs1) for _ in 1:10]
        @test all(s -> s == 1, samples1)
        
        # Two elements with clear distribution
        probs2 = Float16[0.0, 1.0]
        samples2 = [Engine.simple_sample(probs2) for _ in 1:100]
        @test all(s -> s == 2, samples2)
        
        # 50/50 distribution - should get both with reasonable frequency
        probs3 = Float16[0.5, 0.5]
        samples3 = [Engine.simple_sample(probs3) for _ in 1:1000]
        count1 = count(==(1), samples3)
        count2 = count(==(2), samples3)
        @test 400 < count1 < 600  # Should be roughly 50%
        @test 400 < count2 < 600
        
        # Three elements with unequal distribution
        probs4 = Float16[0.1, 0.3, 0.6]
        samples4 = [Engine.simple_sample(probs4) for _ in 1:1000]
        counts = [count(==(i), samples4) for i in 1:3]
        @test counts[1] < counts[2] < counts[3]  # Order should match probabilities
    end

    @testset "sample - temperature=0 (argmax)" begin
        # Basic argmax
        logits1 = Float16[1.0, 5.0, 2.0, 4.0]
        result1 = Engine.sample(logits1, Float16(0.0), Float16(1.0))
        @test result1 == 2  # Index of max (5.0)
        
        # Negative values
        logits2 = Float16[-10.0, -5.0, -1.0]
        result2 = Engine.sample(logits2, Float16(0.0), Float16(1.0))
        @test result2 == 3  # Index of max (-1.0)
        
        # Duplicate max - should return first occurrence
        logits3 = Float16[1.0, 5.0, 2.0, 5.0]
        result3 = Engine.sample(logits3, Float16(0.0), Float16(1.0))
        @test result3 == 2  # First occurrence of max
        
        # Single element
        logits4 = Float16[42.0]
        result4 = Engine.sample(logits4, Float16(0.0), Float16(1.0))
        @test result4 == 1
        
        # All equal values
        logits5 = Float16[3.0, 3.0, 3.0]
        result5 = Engine.sample(logits5, Float16(0.0), Float16(1.0))
        @test result5 == 1  # First index
        
 # Very large values (within Float16 range)
 logits6 = Float16[100.0, 500.0, 1000.0]
 result6 = Engine.sample(logits6, Float16(0.0), Float16(1.0))
 @test result6 == 3
        
        # Very small/negative values
        logits7 = Float16[-1e4, -1e5, -1e3]
        result7 = Engine.sample(logits7, Float16(0.0), Float16(1.0))
        @test result7 == 3
    end

    @testset "sample - top-k sampling" begin
        Random.seed!(42)
        
        # Top-2: only top 2 should ever be sampled
        logits = Float16[1.0, 10.0, 2.0, 9.0]  # Top-2 are indices 2 and 4
        samples = [Engine.sample(copy(logits), Float16(0.7), Float16(1.0), 2) for _ in 1:100]
        @test all(s -> s == 2 || s == 4, samples)
        
        # Top-1 should behave like argmax
        logits2 = Float16[1.0, 5.0, 3.0]
        samples2 = [Engine.sample(copy(logits2), Float16(0.7), Float16(1.0), 1) for _ in 1:50]
        @test all(s -> s == 2, samples2)  # Always picks max
        
        # Top-k larger than array - should not crash
        logits3 = Float16[1.0, 2.0, 3.0]
        result = Engine.sample(copy(logits3), Float16(0.5), Float16(1.0), 100)
        @test 1 <= result <= 3
        
        # Top-k = 0 should sample from all
        logits4 = Float16[10.0, 1.0, 1.0, 1.0]
        result4 = Engine.sample(copy(logits4), Float16(0.5), Float16(1.0), 0)
        @test 1 <= result4 <= 4
    end

    @testset "sample - top-p (nucleus) sampling" begin
        Random.seed!(42)
        
        # Top-p = 1.0 should sample from all (no truncation)
        logits = Float16[1.0, 2.0, 3.0, 4.0]
        samples = [Engine.sample(copy(logits), Float16(0.7), Float16(1.0), 0) for _ in 1:100]
        @test all(s -> 1 <= s <= 4, samples)
        
        # Top-p = 0.5 should only sample from highest probability tokens
        # With logits [1, 2, 3, 4], softmax probs are roughly [0.09, 0.12, 0.17, 0.62]
        # Top-50% should mostly be token 4
        logits2 = Float16[1.0, 2.0, 3.0, 10.0]  # Token 4 dominates
        samples2 = [Engine.sample(copy(logits2), Float16(0.5), Float16(0.5), 0) for _ in 1:100]
        count_4 = count(==(4), samples2)
        @test count_4 > 50  # Should heavily favor token 4
        
        # Edge case: top_p very small
        logits3 = Float16[1.0, 100.0, 2.0]  # Token 2 dominates massively
        samples3 = [Engine.sample(copy(logits3), Float16(0.5), Float16(0.01), 0) for _ in 1:50]
        @test all(s -> s == 2, samples3)  # Should always pick the dominant token
    end

    @testset "sample - temperature scaling" begin
        Random.seed!(42)
        
        # High temperature (more random)
        logits = Float16[1.0, 10.0, 2.0]
        samples_high = [Engine.sample(copy(logits), Float16(2.0), Float16(1.0), 0) for _ in 1:200]
        # With high temp, should see more variety
        unique_high = length(unique(samples_high))
        @test unique_high >= 2
        
        # Low temperature (more deterministic)
        samples_low = [Engine.sample(copy(logits), Float16(0.1), Float16(1.0), 0) for _ in 1:200]
        count_max = count(==(2), samples_low)  # Token 2 has max logit
        @test count_max > 150  # Should heavily favor max
        
        # Temperature = 1.0 (standard softmax)
        samples_std = [Engine.sample(copy(logits), Float16(1.0), Float16(1.0), 0) for _ in 1:200]
        @test all(s -> 1 <= s <= 3, samples_std)
    end

    @testset "sample - non-finite logit handling" begin
        # NaN handling
        logits_nan = Float16[1.0, NaN, 2.0]
        result1 = Engine.sample(logits_nan, Float16(0.0), Float16(1.0))
        @test result1 == 3  # Should pick max among finite values
        
        # Inf handling
        logits_inf = Float16[1.0, Inf, 2.0]
        result2 = Engine.sample(logits_inf, Float16(0.0), Float16(1.0))
        @test result2 == 3  # Should handle gracefully
        
        # -Inf handling
        logits_ninf = Float16[-Inf, 1.0, 2.0]
        result3 = Engine.sample(logits_ninf, Float16(0.0), Float16(1.0))
        @test result3 == 3  # Should pick max among finite
    end

    @testset "mask_and_sample" begin
        Random.seed!(42)
        
        # Basic masking
        logits = Float16[1.0, 5.0, 3.0, 4.0]
        pad_ids = Int[2]  # Mask index 2 (the max)
        result = Engine.mask_and_sample(logits, pad_ids, Float16(0.0), Float16(1.0))
        @test result != 2  # Should not return masked index with temp=0
        
        # Multiple masked tokens
        logits2 = Float16[5.0, 4.0, 3.0, 2.0, 1.0]
        pad_ids2 = Int[1, 2, 3]  # Mask top 3
        result2 = Engine.mask_and_sample(logits2, pad_ids2, Float16(0.0), Float16(1.0))
        @test result2 == 4  # Should return 4th index (value 2.0)
        
        # Empty mask list - should work normally
        logits3 = Float16[1.0, 5.0, 3.0]
        result3 = Engine.mask_and_sample(logits3, Int[], Float16(0.0), Float16(1.0))
        @test result3 == 2
        
        # Invalid pad_ids (out of bounds) - should handle gracefully
        logits4 = Float16[1.0, 2.0, 3.0]
        result4 = Engine.mask_and_sample(logits4, Int[0, 100], Float16(0.0), Float16(1.0))
        @test 1 <= result4 <= 3
        
        # All tokens masked - should fallback to argmax of remaining
        logits5 = Float16[1.0, 2.0, 3.0]
        pad_ids5 = Int[1, 2, 3]
        result5 = Engine.mask_and_sample(logits5, pad_ids5, Float16(0.0), Float16(1.0))
        @test 1 <= result5 <= 3
    end

    @testset "cpu_sample_from_scaled" begin
        Random.seed!(42)
        
        # Basic sampling from already scaled logits
        scaled = Float16[0.1, 0.3, 0.6]  # Already scaled/probability-like
        result = Engine.cpu_sample_from_scaled(scaled, Float16(1.0), Float16(1.0), 0)
        @test 1 <= result <= 3
        
        # With top-k
        scaled2 = Float16[0.1, 0.4, 0.05, 0.45]
        samples2 = [Engine.cpu_sample_from_scaled(copy(scaled2), Float16(1.0), Float16(1.0), 2) for _ in 1:100]
        # Top-2 should be indices 2 and 4
        @test all(s -> s == 2 || s == 4, samples2)
        
 # With top-p
 scaled3 = Float16[0.01, 0.01, 0.01, 0.97]
 samples3 = [Engine.cpu_sample_from_scaled(copy(scaled3), Float16(1.0), Float16(0.5), 0) for _ in 1:50]
 # Should mostly sample index 4 due to high probability
 count_4 = count(==(4), samples3)
 @test count_4 >= 40 # Should heavily favor index 4
    end

    @testset "sample - combined top-k and top-p" begin
        Random.seed!(42)
        
        # Both top-k and top-p applied
        logits = Float16[1.0, 10.0, 2.0, 9.0, 3.0, 8.0]  # Sorted: 10, 9, 8, 3, 2, 1
        samples = [Engine.sample(copy(logits), Float16(0.5), Float16(0.8), 3) for _ in 1:100]
        # Top-3 are indices 2, 4, 6 (values 10, 9, 8)
        @test all(s -> s == 2 || s == 4 || s == 6, samples)
    end

end