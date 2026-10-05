using Test
using Inferno
using Random

const CommonOps = Inferno.CommonOps

@testset "CommonOps softmax sampler" begin
    @testset "oversized reusable scratch" begin
        scratch = CommonOps.create_softmax_scratch(16)
        fill!(scratch.logits_buf, Float32(17))
        fill!(scratch.exp_probs, Float32(19))
        fill!(scratch.indices, -7)
        fill!(scratch.keep_mask, true)

        cases = (
            Float32[0, 1, 2],
            Float32[3, 2, 1, 0, -1, -2, -3],
            Float32[4, -Inf32, 2],
            Float32[0, 1],
        )
        for logits in cases
            original = copy(logits)
            Random.seed!(100 + length(logits))
            token = CommonOps.softmax_sample_scratch!(logits, scratch;
                temperature=Float32(0.7), top_p=Float32(0.8), top_k=2, min_p=Float32(0.1))
            @test 1 <= token <= length(logits)
            @test logits == original
        end

        @test scratch.indices[9:end] == fill(-7, 8)
        @test scratch.keep_mask[9:end] == fill(true, 8)
        @test scratch.logits_buf[9:end] == fill(Float32(17), 8)
        @test scratch.exp_probs[9:end] == fill(Float32(19), 8)
    end

    @testset "input logits are preserved" begin
        logits = Float32[1, 2, 3, 4]
        original = copy(logits)
        scratch = CommonOps.create_softmax_scratch(length(logits))
        Random.seed!(7)
        for _ in 1:20
            CommonOps.softmax_sample_scratch!(logits, scratch;
                temperature=Float32(0.25), top_p=Float32(0.9), top_k=3, min_p=Float32(0.05))
            @test logits == original
        end
    end

    @testset "allocation-free reusable path" begin
        logits = Float32[0, 1, 2, 3, 4, 5]
        scratch = CommonOps.create_softmax_scratch(length(logits))
        Random.seed!(11)
        CommonOps.softmax_sample_scratch!(logits, scratch;
            temperature=Float32(0.7), top_p=Float32(0.8), top_k=4, min_p=Float32(0.1))

        GC.gc()
        allocations = @allocated CommonOps.softmax_sample_scratch!(logits, scratch;
            temperature=Float32(0.7), top_p=Float32(0.8), top_k=4, min_p=Float32(0.1))
        @test allocations == 0

        CommonOps.softmax_sample(logits; temperature=0.0f0)
        GC.gc()
        greedy_allocations = @allocated CommonOps.softmax_sample(logits; temperature=0.0f0)
        @test greedy_allocations == 0
    end

    @testset "tiny positive temperature" begin
        logits = Float32[0, 1, 2]
        original = copy(logits)
        scratch = CommonOps.create_softmax_scratch(length(logits))
        for _ in 1:20
            @test CommonOps.softmax_sample_scratch!(logits, scratch;
                temperature=Float32(1e-30), top_k=1) == 3
        end
        @test logits == original
    end

    @testset "non-finite logits" begin
        logits = Float32[-Inf32, 0, -Inf32, 1]
        scratch = CommonOps.create_softmax_scratch(length(logits))
        Random.seed!(23)
        samples = [CommonOps.softmax_sample_scratch!(logits, scratch; top_k=2) for _ in 1:100]
        @test all(token -> token == 2 || token == 4, samples)
        @test length(unique(samples)) == 2

        @test CommonOps.softmax_sample_scratch!(Float32[-Inf32, -Inf32], scratch) == 1
        @test CommonOps.softmax_sample(Float32[-Inf32, -Inf32]; top_p=0.5f0, top_k=1) == 1

        nonfinite_errors = (
            () -> CommonOps.softmax_sample_scratch!(Float32[1, Inf32], scratch),
            () -> CommonOps.softmax_sample_scratch!(Float32[1, NaN32], scratch),
        )
        for make_call in nonfinite_errors
            @test_throws ArgumentError make_call()
        end
    end

    @testset "parameter validation and empty logits" begin
        logits = Float32[0, 1]
        scratch = CommonOps.create_softmax_scratch(length(logits))
        invalid_calls = (
            () -> CommonOps.softmax_sample(logits; temperature=-1.0f0),
            () -> CommonOps.softmax_sample(logits; temperature=NaN32),
            () -> CommonOps.softmax_sample(logits; temperature=Inf32),
            () -> CommonOps.softmax_sample(logits; top_p=-0.1f0),
            () -> CommonOps.softmax_sample(logits; top_p=1.1f0),
            () -> CommonOps.softmax_sample(logits; top_p=NaN32),
            () -> CommonOps.softmax_sample(logits; top_k=-1),
            () -> CommonOps.softmax_sample(logits; min_p=-0.1f0),
            () -> CommonOps.softmax_sample(logits; min_p=1.1f0),
            () -> CommonOps.softmax_sample(logits; min_p=NaN32),
        )
        for make_call in invalid_calls
            @test_throws ArgumentError make_call()
        end

        @test_throws ArgumentError CommonOps.softmax_sample(Float32[])
        @test_throws ArgumentError CommonOps.softmax_sample_scratch!(Float32[], scratch)
    end

    @testset "filter boundaries" begin
        @test CommonOps.softmax_sample(Float32[0, 1, 2]; top_p=0.0f0) == 3

        Random.seed!(31)
        top_k_logits = Float32[0, 1, 2, 3]
        top_k_samples = [CommonOps.softmax_sample(top_k_logits; top_k=2) for _ in 1:200]
        @test all(token -> token == 3 || token == 4, top_k_samples)
        @test length(unique(top_k_samples)) == 2

        top_k_disabled_logits = Float32[0, 0, 0, 0]
        top_k_disabled_samples = [CommonOps.softmax_sample(top_k_disabled_logits;
            top_k=length(top_k_disabled_logits)) for _ in 1:200]
        @test Set(top_k_disabled_samples) == Set(1:4)

        Random.seed!(37)
        top_p_boundary_logits = log.(Float32[0.2, 0.31, 0.49])
        top_p_boundary_samples = [CommonOps.softmax_sample(top_p_boundary_logits;
            top_p=0.5f0) for _ in 1:200]
        @test all(token -> token == 2 || token == 3, top_p_boundary_samples)
        @test length(unique(top_p_boundary_samples)) == 2

        Random.seed!(41)
        min_p_boundary_logits = Float32[0, 0, 0, 0]
        min_p_boundary_samples = [CommonOps.softmax_sample(min_p_boundary_logits;
            min_p=1.0f0) for _ in 1:400]
        @test Set(min_p_boundary_samples) == Set(1:4)
    end
end
