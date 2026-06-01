# Unit Tests for GGUF Module
#
# Tests for:
# - String reading with length limits
# - Value type reading for all GGUF types
# - GGUF file parsing
# - Tensor info extraction
# - Magic number and version validation

using Test
using Inferno
using Inferno.GGUF

@testset "GGUF Module Tests" begin

    @testset "Constants" begin
        @test GGUF.GGUF_MAGIC == 0x46554747  # "GGUF" in little-endian
        @test GGUF.GGUF_VERSION == 3
    end

    @testset "GGUFValueType enum" begin
        # Test all enum values exist
        @test GGUF.GGUF_TYPE_UINT8 == GGUF.GGUFValueType(0)
        @test GGUF.GGUF_TYPE_INT8 == GGUF.GGUFValueType(1)
        @test GGUF.GGUF_TYPE_UINT16 == GGUF.GGUFValueType(2)
        @test GGUF.GGUF_TYPE_INT16 == GGUF.GGUFValueType(3)
        @test GGUF.GGUF_TYPE_UINT32 == GGUF.GGUFValueType(4)
        @test GGUF.GGUF_TYPE_INT32 == GGUF.GGUFValueType(5)
        @test GGUF.GGUF_TYPE_FLOAT32 == GGUF.GGUFValueType(6)
        @test GGUF.GGUF_TYPE_BOOL == GGUF.GGUFValueType(7)
        @test GGUF.GGUF_TYPE_STRING == GGUF.GGUFValueType(8)
        @test GGUF.GGUF_TYPE_ARRAY == GGUF.GGUFValueType(9)
        @test GGUF.GGUF_TYPE_UINT64 == GGUF.GGUFValueType(10)
        @test GGUF.GGUF_TYPE_INT64 == GGUF.GGUFValueType(11)
        @test GGUF.GGUF_TYPE_FLOAT64 == GGUF.GGUFValueType(12)
        
        # Test conversion from UInt32
        @test GGUF.GGUFValueType(0) isa GGUF.GGUFValueType
        @test GGUF.GGUFValueType(12) isa GGUF.GGUFValueType
    end

    @testset "GGMLType enum" begin
        # Test common types exist
        @test GGUF.GGML_TYPE_F32 == GGUF.GGMLType(0)
        @test GGUF.GGML_TYPE_F16 == GGUF.GGMLType(1)
        @test GGUF.GGML_TYPE_Q4_0 == GGUF.GGMLType(2)
        @test GGUF.GGML_TYPE_Q8_0 == GGUF.GGMLType(8)
        @test GGUF.GGML_TYPE_IQ2_XXS == GGUF.GGMLType(16)
        @test GGUF.GGML_TYPE_IQ4_NL == GGUF.GGMLType(20)
    end

    @testset "read_string - normal cases" begin
        # Short string
        io = IOBuffer()
        write(io, UInt64(5))  # length
        write(io, "Hello")
        seekstart(io)
        result = GGUF.read_string(io)
        @test result == "Hello"
        
        # Empty string
        io2 = IOBuffer()
        write(io2, UInt64(0))
        seekstart(io2)
        result2 = GGUF.read_string(io2)
        @test result2 == ""
        
 # String with special characters
 io3 = IOBuffer()
 str3 = "Hello 世界!"
 write(io3, UInt64(ncodeunits(str3))) # Use byte length, not character length
 write(io3, str3)
 seekstart(io3)
 result3 = GGUF.read_string(io3)
 @test result3 == "Hello 世界!"
        
        # Maximum allowed length (1MB - 1)
        io4 = IOBuffer()
        long_str = "a" ^ (1048575)
        write(io4, UInt64(length(long_str)))
        write(io4, long_str)
        seekstart(io4)
        result4 = GGUF.read_string(io4)
        @test length(result4) == 1048575
        
        # Exactly at limit (1MB)
        io5 = IOBuffer()
        long_str2 = "b" ^ 1048576
        write(io5, UInt64(length(long_str2)))
        write(io5, long_str2)
        seekstart(io5)
        result5 = GGUF.read_string(io5)
        @test length(result5) == 1048576
    end

    @testset "read_string - security limits" begin
        # Over limit (1MB + 1) - should throw
        io = IOBuffer()
        write(io, UInt64(1048577))
        seekstart(io)
        @test_throws ErrorException GGUF.read_string(io)
        
        # Very large length (max UInt64) - should throw
        io2 = IOBuffer()
        write(io2, UInt64(typemax(UInt64)))
        seekstart(io2)
        @test_throws ErrorException GGUF.read_string(io2)
        
        # Large but not astronomical
        io3 = IOBuffer()
        write(io3, UInt64(100 * 1048576))  # 100MB
        seekstart(io3)
        @test_throws ErrorException GGUF.read_string(io3)
    end

    @testset "read_value - integer types" begin
        # UINT8
        io = IOBuffer()
        write(io, UInt8(42))
        seekstart(io)
        result = GGUF.read_value(io, GGUF.GGUF_TYPE_UINT8)
        @test result == 42
        @test isa(result, UInt8)
        
        # INT8
        io2 = IOBuffer()
        write(io2, Int8(-42))
        seekstart(io2)
        result2 = GGUF.read_value(io2, GGUF.GGUF_TYPE_INT8)
        @test result2 == -42
        @test isa(result2, Int8)
        
        # UINT16
        io3 = IOBuffer()
        write(io3, UInt16(1000))
        seekstart(io3)
        result3 = GGUF.read_value(io3, GGUF.GGUF_TYPE_UINT16)
        @test result3 == 1000
        @test isa(result3, UInt16)
        
        # INT16
        io4 = IOBuffer()
        write(io4, Int16(-1000))
        seekstart(io4)
        result4 = GGUF.read_value(io4, GGUF.GGUF_TYPE_INT16)
        @test result4 == -1000
        @test isa(result4, Int16)
        
        # UINT32
        io5 = IOBuffer()
        write(io5, UInt32(100000))
        seekstart(io5)
        result5 = GGUF.read_value(io5, GGUF.GGUF_TYPE_UINT32)
        @test result5 == 100000
        @test isa(result5, UInt32)
        
        # INT32
        io6 = IOBuffer()
        write(io6, Int32(-100000))
        seekstart(io6)
        result6 = GGUF.read_value(io6, GGUF.GGUF_TYPE_INT32)
        @test result6 == -100000
        @test isa(result6, Int32)
        
        # UINT64
        io7 = IOBuffer()
        write(io7, UInt64(10000000000))
        seekstart(io7)
        result7 = GGUF.read_value(io7, GGUF.GGUF_TYPE_UINT64)
        @test result7 == 10000000000
        @test isa(result7, UInt64)
        
        # INT64
        io8 = IOBuffer()
        write(io8, Int64(-10000000000))
        seekstart(io8)
        result8 = GGUF.read_value(io8, GGUF.GGUF_TYPE_INT64)
        @test result8 == -10000000000
        @test isa(result8, Int64)
    end

    @testset "read_value - float types" begin
        # FLOAT32
        io = IOBuffer()
        write(io, Float32(3.14159))
        seekstart(io)
        result = GGUF.read_value(io, GGUF.GGUF_TYPE_FLOAT32)
        @test isapprox(result, 3.14159, atol=1e-5)
        @test isa(result, Float32)
        
        # FLOAT64
        io2 = IOBuffer()
        write(io2, Float64(3.14159265358979))
        seekstart(io2)
        result2 = GGUF.read_value(io2, GGUF.GGUF_TYPE_FLOAT64)
        @test isapprox(result2, 3.14159265358979, atol=1e-14)
        @test isa(result2, Float64)
    end

    @testset "read_value - bool type" begin
        # True
        io = IOBuffer()
        write(io, true)
        seekstart(io)
        result = GGUF.read_value(io, GGUF.GGUF_TYPE_BOOL)
        @test result == true
        @test isa(result, Bool)
        
        # False
        io2 = IOBuffer()
        write(io2, false)
        seekstart(io2)
        result2 = GGUF.read_value(io2, GGUF.GGUF_TYPE_BOOL)
        @test result2 == false
        @test isa(result2, Bool)
    end

    @testset "read_value - string type" begin
        io = IOBuffer()
        write(io, UInt64(5))
        write(io, "World")
        seekstart(io)
        result = GGUF.read_value(io, GGUF.GGUF_TYPE_STRING)
        @test result == "World"
        @test isa(result, String)
    end

    @testset "read_value - array type" begin
        # Array of integers
        io = IOBuffer()
        write(io, UInt32(5))  # element type: INT32
        write(io, UInt64(3))  # length
        write(io, Int32(1))
        write(io, Int32(2))
        write(io, Int32(3))
        seekstart(io)
        result = GGUF.read_value(io, GGUF.GGUF_TYPE_ARRAY)
        @test result == [1, 2, 3]
        @test isa(result, Vector)
        
        # Array of strings
        io2 = IOBuffer()
        write(io2, UInt32(8))  # element type: STRING
        write(io2, UInt64(2))  # length
        write(io2, UInt64(2))
        write(io2, "Hi")
        write(io2, UInt64(5))
        write(io2, "World")
        seekstart(io2)
        result2 = GGUF.read_value(io2, GGUF.GGUF_TYPE_ARRAY)
        @test result2 == ["Hi", "World"]
        
        # Empty array
        io3 = IOBuffer()
        write(io3, UInt32(5))  # element type: INT32
        write(io3, UInt64(0))  # length
        seekstart(io3)
        result3 = GGUF.read_value(io3, GGUF.GGUF_TYPE_ARRAY)
        @test isempty(result3)
    end

    @testset "read_value - unknown type" begin
        io = IOBuffer()
        # Create an invalid type value
        seekstart(io)
        @test_throws ErrorException GGUF.read_value(io, reinterpret(GGUF.GGUFValueType, 999 % UInt32))
    end

    @testset "TensorInfo struct" begin
        dims = [1024, 4096]
        info = GGUF.TensorInfo("test.weight", dims, GGUF.GGML_TYPE_F16, 0)
        
        @test info.name == "test.weight"
        @test info.dimensions == dims
        @test info.type == GGUF.GGML_TYPE_F16
        @test info.offset == 0
    end

    @testset "GGUFFile struct" begin
        metadata = Dict{String, Any}("key" => "value")
        tensors = Dict{String, GGUF.TensorInfo}()
        file = GGUF.GGUFFile(metadata, tensors, 0, UInt8[])
        
        @test file.metadata == metadata
        @test file.tensors == tensors
        @test file.data_offset == 0
        @test isempty(file.tensor_data)
    end

    @testset "read_gguf - invalid file" begin
        # Create a temporary file with invalid magic
        mktemp() do path, io
            write(io, UInt32(0x12345678))  # Invalid magic
            close(io)
            @test_throws ErrorException("Not a valid GGUF file") GGUF.read_gguf(path)
        end
    end

    @testset "read_gguf - valid file structure" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = GGUF.read_gguf(MODEL_PATH)
            
            # Basic structure checks
            @test isa(file, GGUF.GGUFFile)
            @test isa(file.metadata, Dict{String, Any})
            @test isa(file.tensors, Dict{String, GGUF.TensorInfo})
            @test file.data_offset > 0
            @test !isempty(file.tensor_data)
            
            # Metadata checks
            @test haskey(file.metadata, "general.architecture")
            @test file.metadata["general.architecture"] isa String
            
            # Tensor checks
            @test !isempty(file.tensors)
            
            # Check tensor info structure
            for (name, info) in file.tensors
                @test info.name == name
                @test !isempty(info.dimensions)
                @test info.type isa GGUF.GGMLType
                @test info.offset >= 0
            end
        end
    end

    @testset "get_tensor" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = GGUF.read_gguf(MODEL_PATH)
            
            # Get existing tensor
            info = GGUF.get_tensor(file, "token_embd.weight")
            @test isa(info, GGUF.TensorInfo)
            @test info.name == "token_embd.weight"
            
            # Non-existent tensor
            @test_throws ErrorException GGUF.get_tensor(file, "nonexistent.tensor")
        end
    end

end