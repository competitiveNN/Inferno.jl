module GGUF

import Mmap

# GGUF Magic and Version
const GGUF_MAGIC = 0x46554747
const GGUF_VERSION = 3

@enum GGUFValueType::UInt32 begin
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
end

@enum GGMLType::UInt32 begin
    GGML_TYPE_F32     = 0
    GGML_TYPE_F16     = 1
    GGML_TYPE_Q4_0    = 2
    GGML_TYPE_Q4_1    = 3
    GGML_TYPE_Q5_0    = 6
    GGML_TYPE_Q5_1    = 7
    GGML_TYPE_Q8_0    = 8
    GGML_TYPE_Q8_1    = 9
    GGML_TYPE_Q2_K    = 10
    GGML_TYPE_Q3_K    = 11
    GGML_TYPE_Q4_K    = 12
    GGML_TYPE_Q5_K    = 13
    GGML_TYPE_Q6_K    = 14
    GGML_TYPE_Q8_K    = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S   = 19
    GGML_TYPE_IQ4_NL  = 20
    GGML_TYPE_IQ3_S   = 21
    GGML_TYPE_IQ2_S   = 22
    GGML_TYPE_IQ4_XS  = 23
    GGML_TYPE_I8      = 24
    GGML_TYPE_I16     = 25
    GGML_TYPE_I32     = 26
    GGML_TYPE_I64     = 27
    GGML_TYPE_F64     = 28
    GGML_TYPE_IQ1_M   = 29
    GGML_TYPE_BF16    = 30
    GGML_TYPE_Q4_0_4_4 = 31
    GGML_TYPE_Q4_0_4_8 = 32
    GGML_TYPE_Q4_0_8_8 = 33
    GGML_TYPE_TQ1_0   = 34
    GGML_TYPE_TQ2_0   = 35
end

struct TensorInfo
    name::String
    dimensions::Vector{UInt64}
    type::GGMLType
    offset::UInt64
end

struct GGUFFile
    metadata::Dict{String, Any}
    tensors::Dict{String, TensorInfo}
    data_offset::UInt64
    tensor_data::Vector{UInt8} # mmapped array containing all data
end

function read_string(io::IO)
    len = read(io, UInt64)
    if len > 1048576 # 1MB limit to prevent OOM DOS
        error("String length exceeds maximum allowed length of 1MB")
    end
    String(read(io, len))
end

function read_value(io::IO, type::GGUFValueType)
    if type == GGUF_TYPE_UINT8
        return read(io, UInt8)
    elseif type == GGUF_TYPE_INT8
        return read(io, Int8)
    elseif type == GGUF_TYPE_UINT16
        return read(io, UInt16)
    elseif type == GGUF_TYPE_INT16
        return read(io, Int16)
    elseif type == GGUF_TYPE_UINT32
        return read(io, UInt32)
    elseif type == GGUF_TYPE_INT32
        return read(io, Int32)
    elseif type == GGUF_TYPE_FLOAT32
        return read(io, Float32)
    elseif type == GGUF_TYPE_UINT64
        return read(io, UInt64)
    elseif type == GGUF_TYPE_INT64
        return read(io, Int64)
    elseif type == GGUF_TYPE_FLOAT64
        return read(io, Float64)
    elseif type == GGUF_TYPE_BOOL
        return read(io, Bool)
    elseif type == GGUF_TYPE_STRING
        return read_string(io)
    elseif type == GGUF_TYPE_ARRAY
        atype = GGUFValueType(read(io, UInt32))
        len = read(io, UInt64)
        return [read_value(io, atype) for _ in 1:len]
    else
        error("Unknown GGUF type: $type")
    end
end

export read_gguf
function read_gguf(path::AbstractString)
    io = open(path, "r")

    magic = read(io, UInt32)
    if magic != GGUF_MAGIC
        error("Not a valid GGUF file")
    end

    version = read(io, UInt32)
    if version != GGUF_VERSION
        @warn "Unsupported GGUF version" version=version expected=GGUF_VERSION
    end

    tensor_count = read(io, UInt64)
    kv_count = read(io, UInt64)

    metadata = Dict{String, Any}()
    for _ in 1:kv_count
        key = read_string(io)
        vtype = GGUFValueType(read(io, UInt32))
        metadata[key] = read_value(io, vtype)
    end

    tensors = Dict{String, TensorInfo}()
    for _ in 1:tensor_count
        name = read_string(io)
        ndim = read(io, UInt32)
        dims = [read(io, UInt64) for _ in 1:ndim]
        type = GGMLType(read(io, UInt32))
        offset = read(io, UInt64)
        tensors[name] = TensorInfo(name, dims, type, offset)
    end

    alignment = get(metadata, "general.alignment", 32)
    padding = alignment - (position(io) % alignment)
    if padding != alignment && padding > 0
        read(io, padding)
    end

    data_offset = position(io)
    close(io)

    # Memory map the tensor data
    mapped_data = Mmap.mmap(path)

    return GGUFFile(metadata, tensors, data_offset, mapped_data)
end

export get_tensor, get_tensor_raw_bytes
function get_tensor(file::GGUFFile, name::String)
    if !haskey(file.tensors, name)
        error("Tensor not found: $name")
    end

    info = file.tensors[name]
    start_idx = file.data_offset + info.offset + 1

    # Calculate bytes based on type and dimensions, returning raw bytes
    # For simplicity, assuming f32 or f16 for now to get length, complex types need block sizing
    # It would be better to return a pointer or a view of the bytes
    return info
end

"""
    get_tensor_raw_bytes(file::GGUFFile, name::String)

Get raw tensor data bytes from the mapped memory.
This is useful for dequantization or direct GPU upload.
"""
function get_tensor_raw_bytes(file::GGUFFile, name::String)
    if !haskey(file.tensors, name)
        error("Tensor not found: $name")
    end

    info = file.tensors[name]
    start_offset = file.data_offset + info.offset
    
    # Calculate size in bytes based on type and dimensions
    # For Q4_K types, this is complex (5 bytes per 32 values). Simplifying for now.
    # We return a view of the raw bytes.
    total_elements = prod(info.dimensions)
    
    # Approximate: assume 4 bytes per element (worst case F32)
    # For quantized types, we'd need to use dequantization logic
    bytes_per_element = 4  # Upper bound
    raw_bytes = @view file.tensor_data[start_offset + 1 : start_offset + total_elements * bytes_per_element]
    
    return raw_bytes
end

end # module
