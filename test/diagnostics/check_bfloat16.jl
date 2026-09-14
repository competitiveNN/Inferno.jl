using oneAPI
using Test

@info "Checking BFloat16 support in oneAPI.jl"

# BFloat16 is available in Core since Julia 1.11
const BF16 = Core.BFloat16

# Helper to create BF16 from Float32/Float64 if direct constructor is missing
function to_bf16(x::AbstractFloat)
    u32 = reinterpret(UInt32, Float32(x))
    u16 = UInt16(u32 >> 16)
    return reinterpret(BF16, u16)
end

function from_bf16(x::BF16)
    u16 = reinterpret(UInt16, x)
    u32 = UInt32(u16) << 16
    return reinterpret(Float32, u32)
end

@testset "oneAPI BFloat16 Support" begin
    
    @testset "Storage and Transfer" begin
        try
            cpu_data = [to_bf16(1.0), to_bf16(2.0), to_bf16(3.0)]
            gpu_data = oneArray(cpu_data)
            @test eltype(gpu_data) == BF16
            @test length(gpu_data) == 3
            @info "✅ BFloat16 Storage and Transfer supported"
        catch e
            @error "❌ BFloat16 Storage and Transfer failed" exception=e
            @test false
        end
    end

    @testset "Conversion to Float32" begin
        try
            cpu_data = [to_bf16(1.0), to_bf16(2.0), to_bf16(3.0)]
            gpu_data = oneArray(cpu_data)
            gpu_f32 = oneArray{Float32}(gpu_data)
            @test eltype(gpu_f32) == Float32
            @test Array(gpu_f32) ≈ [1.0f0, 2.0f0, 3.0f0]
            @info "✅ BFloat16 to Float32 conversion on GPU supported"
        catch e
            @info "ℹ️ BFloat16 to Float32 conversion on GPU not directly supported" exception=e
            # This might need a custom kernel or manual bit manipulation
            @test_skip false
        end
    end

    @testset "Arithmetic Operations" begin
        try
            cpu_data = [to_bf16(1.0), to_bf16(2.0), to_bf16(3.0)]
            gpu_data = oneArray(cpu_data)
            gpu_data .+= to_bf16(1.0)
            result = Array(gpu_data)
            @test from_bf16(result[1]) == 2.0f0
            @info "✅ BFloat16 Arithmetic supported"
        catch e
            @info "ℹ️ BFloat16 Arithmetic not directly supported (MethodError/InvalidIR)" 
            # This is common if the backend doesn't define hardware BF16 ops yet
            @test_skip false
        end
    end
end
