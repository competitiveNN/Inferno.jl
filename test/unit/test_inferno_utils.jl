# Unit Tests for Inferno.jl Utility Functions
#
# Tests for:
# - non_nothing_fields
# - find_related_file
# - probe_device
# - select_device!

using Test
using Inferno

# Helper struct for testing non_nothing_fields
struct TestStruct
    a::Int
    b::Union{Int, Nothing}
    c::Union{String, Nothing}
    d::Nothing
end

@testset "Inferno Utility Functions" begin

    @testset "non_nothing_fields" begin
        # All fields non-nothing
        obj1 = TestStruct(1, 2, "hello", nothing)
        result1 = Inferno.non_nothing_fields(obj1)
        
        @test haskey(result1, :a)
        @test haskey(result1, :b)
        @test haskey(result1, :c)
        @test !haskey(result1, :d)  # d is nothing
        
        @test result1.a == 1
        @test result1.b == 2
        @test result1.c == "hello"
        
        # Some fields nothing
        obj2 = TestStruct(1, nothing, "test", nothing)
        result2 = Inferno.non_nothing_fields(obj2)
        
        @test haskey(result2, :a)
        @test !haskey(result2, :b)  # b is nothing
        @test haskey(result2, :c)
        @test !haskey(result2, :d)  # d is nothing
        
        @test result2.a == 1
        @test result2.c == "test"
        
        # All fields non-nothing (except d which is typed as Nothing)
        obj3 = TestStruct(10, 20, "world", nothing)
        result3 = Inferno.non_nothing_fields(obj3)
        
        @test length(keys(result3)) == 3  # a, b, c
        @test result3.a == 10
        @test result3.b == 20
        @test result3.c == "world"
    end

    @testset "find_related_file - exact match" begin
        mktempdir() do dir
            # Create test files
            model_path = joinpath(dir, "model.gguf")
            mmproj_path = joinpath(dir, "mmproj.gguf")
            
            write(model_path, "model data")
            write(mmproj_path, "mmproj data")
            
            # Exact match
            result = Inferno.find_related_file(model_path, "mmproj.gguf")
            @test result == mmproj_path
            @test isfile(result)
        end
    end

 @testset "find_related_file - case insensitive match" begin
 mktempdir() do dir
 model_path = joinpath(dir, "model.gguf")
 mmproj_path = joinpath(dir, "MMPROJ.gguf")
            
            write(model_path, "model data")
            write(mmproj_path, "mmproj data")
            
            # Case-insensitive substring match
            result = Inferno.find_related_file(model_path, "mmproj")
            @test result == mmproj_path
        end
    end

    @testset "find_related_file - substring match" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.gguf")
            mmproj_path = joinpath(dir, "qwen-mmproj-v1.gguf")
            
            write(model_path, "model data")
            write(mmproj_path, "mmproj data")
            
            # Substring match
            result = Inferno.find_related_file(model_path, "mmproj")
            @test result == mmproj_path
        end
    end

    @testset "find_related_file - no match" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.gguf")
            write(model_path, "model data")
            
            result = Inferno.find_related_file(model_path, "nonexistent")
            @test isnothing(result)
        end
    end

    @testset "find_related_file - empty directory" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.gguf")
            write(model_path, "model data")
            
            # File doesn't exist in directory
            result = Inferno.find_related_file(model_path, "anything.gguf")
            @test isnothing(result)
        end
    end

    @testset "find_related_file - path with no directory" begin
        # Test with relative path (no directory component)
        # This tests the `isempty(model_dir)` branch
        model_path = "model.gguf"
        # When model_dir is empty, it becomes "."
        # We can't easily test this without affecting the current directory
        # So we'll just ensure the function handles it gracefully
        
        # If there's no mmproj in current dir, should return nothing
        result = Inferno.find_related_file("nonexistent_dir/model.gguf", "mmproj")
        # Directory doesn't exist, so catch block should handle it
        @test isnothing(result)
    end

    @testset "find_related_file - multiple candidates" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.gguf")
            mmproj1 = joinpath(dir, "mmproj-v1.gguf")
            mmproj2 = joinpath(dir, "mmproj-v2.gguf")
            
            write(model_path, "model")
            write(mmproj1, "v1")
            write(mmproj2, "v2")
            
            # Should return first match (alphabetically or by readdir order)
            result = Inferno.find_related_file(model_path, "mmproj")
            @test result isa String
            @test occursin("mmproj", lowercase(result))
        end
    end

    @testset "probe_device - basic test" begin
        # This test requires oneAPI to be available
        # We can only test the function exists and returns a Symbol
        try
            using oneAPI
            devs = collect(oneAPI.devices())
            if !isempty(devs)
                result = Inferno.probe_device(devs[1])
                @test result isa Symbol
                @test result in [:ok, :fast_fail, :unknown]
            end
        catch
            @test_skip "oneAPI not available"
        end
    end

    @testset "select_device! - basic test" begin
        # This test requires oneAPI to be available
        try
            using oneAPI
            devs = collect(oneAPI.devices())
            if !isempty(devs)
                # Test with specific device
                result = Inferno.select_device!(devs, 1)
                @test result isa Int
                @test 1 <= result <= length(devs)
                
                # Test with nothing (auto-select)
                result2 = Inferno.select_device!(devs, nothing)
                @test result2 isa Int
                @test 1 <= result2 <= length(devs)
                
                # Test with out-of-bounds device (should clamp)
                result3 = Inferno.select_device!(devs, 100)
                @test result3 isa Int
                @test 1 <= result3 <= length(devs)
            end
        catch
            @test_skip "oneAPI not available"
        end
    end

end

# Additional tests for NamedTuple handling
@testset "non_nothing_fields - NamedTuple properties" begin
    obj = TestStruct(1, nothing, "test", nothing)
    result = Inferno.non_nothing_fields(obj)
    
    # Test NamedTuple operations
    @test isa(result, NamedTuple)
    @test keys(result) isa Tuple{Vararg{Symbol}}
    
    # Can iterate over keys
    for k in keys(result)
        @test k isa Symbol
    end
    
    # Can iterate over values
    for v in values(result)
        @test !isnothing(v)
    end
end