using Test
using Inferno
using Inferno.Safetensors
using Inferno.LoaderCPU
using Inferno.ArrowLake
using BFloat16s

@testset "Safetensors Edge Cases" begin

@testset "BF16 Conversion" begin
    values_f32 = Float32[1.0, 2.0, 3.0, 4.0]
    values_bf16 = BFloat16.(values_f32)
    bits = reinterpret(UInt16, values_bf16)
    bytes = reinterpret(UInt8, bits)
    shape = [2,2]
    tensors = Dict("t" => (1, 3, shape))
    sf = Safetensors.SafetensorsFile("dummy", Dict(), tensors, bytes)
    result = Safetensors.get_tensor(sf, "t")
    expected = Float32[1 2; 3 4]
    @test size(result) == (2,2)
    @test eltype(result) == Float32
    @test result ≈ expected
end

@testset "3D Conv1d - F32" begin
    shape = [4,1,3]
    num = prod(shape)
    vals = Float32.(collect(1:num))
    expected = [Float32((c-1)*3 + k) for c in 1:4, k in 1:3]
    bytes = reinterpret(UInt8, vals)
    tensors = Dict("c" => (1, 1, shape))
    sf = Safetensors.SafetensorsFile("dummy", Dict(), tensors, bytes)
    res = Safetensors.get_tensor(sf, "c")
    @test size(res) == (4,3)
    @test eltype(res) == Float32
    @test res ≈ expected
end

@testset "3D Conv1d - F16" begin
    shape = [2,1,4]
    num = prod(shape)
    vals_f16 = Float16.(Float32.(collect(1:num)))
    expected = [Float32((c-1)*4 + k) for c in 1:2, k in 1:4]
    bytes = reinterpret(UInt8, vals_f16)
    tensors = Dict("c" => (1, 2, shape))
    sf = Safetensors.SafetensorsFile("dummy", Dict(), tensors, bytes)
    res = Safetensors.get_tensor(sf, "c")
    @test size(res) == (2,4)
    @test eltype(res) == Float32
    @test res ≈ expected
end

@testset "3D Conv1d - BF16" begin
    shape = [3,1,2]
    num = prod(shape)
    vals_f32 = Float32.(collect(1:num))
    vals_bf16 = BFloat16.(vals_f32)
    bits = reinterpret(UInt16, vals_bf16)
    bytes = reinterpret(UInt8, bits)
    expected = [Float32((c-1)*2 + k) for c in 1:3, k in 1:2]
    tensors = Dict("c" => (1, 3, shape))
    sf = Safetensors.SafetensorsFile("dummy", Dict(), tensors, bytes)
    res = Safetensors.get_tensor(sf, "c")
    @test size(res) == (3,2)
    @test eltype(res) == Float32
    @test res ≈ expected
end

@testset "Alpha/Beta Transpose" begin
    shape = [2,3]
    vals = Float32[1,2,3,4,5,6]
    bytes = reinterpret(UInt8, vals)
    tensors = Dict("a" => (1, 1, shape))
    sf = Safetensors.SafetensorsFile("dummy", Dict(), tensors, bytes)
    t = Safetensors.get_tensor(sf, "a")
    expected_t = Float32[1 2 3; 4 5 6]
    @test size(t) == (2,3)
    @test t ≈ expected_t
    a = Matrix{Float32}(t')
    expected_a = Float32[1 4; 2 5; 3 6]
    @test size(a) == (3,2)
    @test a ≈ expected_a
end

@testset "maybe_bf16" begin
    w = Float32[1 2; 3 4]
    r = LoaderCPU.maybe_bf16(w, false)
    @test r === w
    r2 = LoaderCPU.maybe_bf16(w, true)
    @test Float32.(r2) ≈ w
    if ArrowLake.has_arrow_lake_features()
        @test eltype(r2) == BFloat16
    else
        @test eltype(r2) == Float32
    end
end

end
