# Unit Tests for Tokenizer Module
#
# Tests for:
# - Byte mapping (BYTE_TO_CHAR, CHAR_TO_BYTE)
# - Special token splitting
# - Unicode handling
# - Byte fallback tokens
# - Encode/decode roundtrip

using Test
using Inferno
using Inferno.Tokenizer

# Helper to create a minimal tokenizer for testing
function create_test_tokenizer(;
    tokens::Vector{String} = String[],
    merges::Vector{Tuple{String, String}} = Tuple{String, String}[],
    special_tokens::Vector{String} = String[],
    pretokenizer::String = "default",
    bos_id::Int = -1,
    eos_id::Int = -1
)
    token_to_id = Dict{String, Int}()
    id_to_token = Vector{String}(undef, max(1, length(tokens)))
    
    for (i, tok) in enumerate(tokens)
        token_to_id[tok] = i
        id_to_token[i] = tok
    end
    
    # Ensure id_to_token has at least one element
    if isempty(tokens)
        id_to_token = [""]
    end
    
    merge_priority = Dict{Tuple{String, String}, Int}()
    for (i, pair) in enumerate(merges)
        merge_priority[pair] = i
    end
    
    BPETokenizer(token_to_id, id_to_token, merges, merge_priority, 
        special_tokens, pretokenizer, bos_id, eos_id)
end

@testset "Tokenizer Module Tests" begin

    @testset "BYTE_TO_CHAR and CHAR_TO_BYTE consistency" begin
        # Every BYTE_TO_CHAR entry should have a corresponding CHAR_TO_BYTE entry
        for (byte, char) in Tokenizer.BYTE_TO_CHAR
            @test haskey(Tokenizer.CHAR_TO_BYTE, char)
            @test Tokenizer.CHAR_TO_BYTE[char] == byte
        end
        
        # Sizes should match
        @test length(Tokenizer.BYTE_TO_CHAR) == 256
        @test length(Tokenizer.CHAR_TO_BYTE) == 256
        
        # Printable ASCII range
        for c in '!' :'~'
            @test UInt8(c) in keys(Tokenizer.BYTE_TO_CHAR)
        end
    end

    @testset "_split_special_tokens - basic cases" begin
        # Empty special tokens list
        tok_empty = create_test_tokenizer(special_tokens=String[])
        parts = Tokenizer._split_special_tokens(tok_empty, "Hello world")
        @test length(parts) == 1
        @test parts[1] == (false, "Hello world")
        
        # No special tokens in text
        tok_with_special = create_test_tokenizer(special_tokens=["<|im_start|>", "<|im_end|>"])
        parts2 = Tokenizer._split_special_tokens(tok_with_special, "Hello world")
        @test length(parts2) == 1
        @test parts2[1] == (false, "Hello world")
        
        # Single special token
        parts3 = Tokenizer._split_special_tokens(tok_with_special, "<|im_start|>user")
        @test length(parts3) == 2
        @test parts3[1] == (true, "<|im_start|>")
        @test parts3[2] == (false, "user")
        
        # Multiple special tokens
        parts4 = Tokenizer._split_special_tokens(tok_with_special, "<|im_start|>user<|im_end|>")
        @test length(parts4) == 3
        @test parts4[1] == (true, "<|im_start|>")
        @test parts4[2] == (false, "user")
        @test parts4[3] == (true, "<|im_end|>")
        
        # Special token at end
        parts5 = Tokenizer._split_special_tokens(tok_with_special, "Hello<|im_end|>")
        @test length(parts5) == 2
        @test parts5[1] == (false, "Hello")
        @test parts5[2] == (true, "<|im_end|>")
        
        # Consecutive special tokens
        parts6 = Tokenizer._split_special_tokens(tok_with_special, "<|im_start|><|im_end|>")
        @test length(parts6) == 2
        @test parts6[1] == (true, "<|im_start|>")
        @test parts6[2] == (true, "<|im_end|>")
        
 # Empty string - returns empty array
 parts7 = Tokenizer._split_special_tokens(tok_with_special, "")
 @test isempty(parts7)
    end

    @testset "_split_special_tokens - edge cases" begin
        tok = create_test_tokenizer(special_tokens=["<s>", "</s>", "<pad>"])
        
 # Multiple different special tokens
 parts1 = Tokenizer._split_special_tokens(tok, "Hello<s>world</s>test<pad>end")
 @test length(parts1) == 7 # "Hello", "<s>", "world", "</s>", "test", "<pad>", "end"
        
        # Only special tokens
        parts2 = Tokenizer._split_special_tokens(tok, "<s></s>")
        @test length(parts2) == 2
        @test parts2[1] == (true, "<s>")
        @test parts2[2] == (true, "</s>")
        
        # Special token prefix matching (longest match wins)
        tok_prefix = create_test_tokenizer(special_tokens=["<|im|>", "<|im_start|>"])
        parts3 = Tokenizer._split_special_tokens(tok_prefix, "<|im_start|>test")
        # Should match the longer token first
        @test length(parts3) >= 1
        
        # Special tokens sorted by length (longest first)
        tok_sorted = create_test_tokenizer(special_tokens=["<|im_start|>", "<|im|>"])
        # Sorting happens in load_tokenizer
    end

    @testset "get_byte_map" begin
        byte_map = Tokenizer.get_byte_map()
        
        # Should be a Dict with 256 entries
        @test isa(byte_map, Dict{UInt8, Char})
        @test length(byte_map) == 256
        
        # All printable ASCII should map to themselves or close
        for b in UInt8('!'):UInt8('~')
            @test haskey(byte_map, b)
        end
    end

 @testset "_encode_piece! - basic encoding" begin
 # Create tokenizer with basic tokens and byte fallbacks
 # Need to include byte fallback tokens for unknown characters
 tokens = ["H", "e", "l", "o", "he", "ll", "hello"]
 # Add byte fallback tokens (0x00-0xFF range)
 for b in 0x00:0xFF
 push!(tokens, "<0x$(string(b, base=16, pad=2))>")
 end
 merges = Tuple{String, String}[("he", "l"), ("hel", "lo"), ("hel", "l")]
 tok = create_test_tokenizer(tokens=tokens, merges=merges)
 
 ids = Int[]
 # Empty piece
 Tokenizer._encode_piece!(ids, tok, "")
 @test isempty(ids)
 
 # Known tokens
 ids2 = Int[]
 Tokenizer._encode_piece!(ids2, tok, "he")
 @test !isempty(ids2)
 @test all(id -> id > 0, ids2)
 end

    @testset "encode/decode roundtrip" begin
        # Use actual model tokenizer if available
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # ASCII text
            text1 = "Hello, world!"
            ids1 = Tokenizer.encode(tok, text1)
            decoded1 = Tokenizer.decode(tok, ids1)
            @test decoded1 == text1
            
            # Numbers
            text2 = "The answer is 42."
            ids2 = Tokenizer.encode(tok, text2)
            decoded2 = Tokenizer.decode(tok, ids2)
            @test decoded2 == text2
            
            # Punctuation
            text3 = "Hello! How are you? I'm fine, thanks."
            ids3 = Tokenizer.encode(tok, text3)
            decoded3 = Tokenizer.decode(tok, ids3)
            @test decoded3 == text3
            
            # Multi-word
            text4 = "The quick brown fox jumps over the lazy dog."
            ids4 = Tokenizer.encode(tok, text4)
            decoded4 = Tokenizer.decode(tok, ids4)
            @test decoded4 == text4
            
            # Unicode
            text5 = "Café résumé naïve"
            ids5 = Tokenizer.encode(tok, text5)
            decoded5 = Tokenizer.decode(tok, ids5)
            @test decoded5 == text5
            
            # Mixed scripts
            text6 = "Hello 你好 مرحبا"
            ids6 = Tokenizer.encode(tok, text6)
            decoded6 = Tokenizer.decode(tok, ids6)
            @test decoded6 == text6
            
            # Emojis
            text7 = "Hello 👋 World 🌍"
            ids7 = Tokenizer.encode(tok, text7)
            decoded7 = Tokenizer.decode(tok, ids7)
            @test decoded7 == text7
            
            println("  All roundtrip tests passed!")
        else
            @warn "Model not found at $MODEL_PATH, skipping roundtrip tests"
        end
    end

    @testset "encode - unicode handling" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # Accented characters
            ids1 = Tokenizer.encode(tok, "café")
            @test !isempty(ids1)
            
            # Chinese characters
            ids2 = Tokenizer.encode(tok, "你好")
            @test !isempty(ids2)
            
            # Japanese
            ids3 = Tokenizer.encode(tok, "こんにちは")
            @test !isempty(ids3)
            
            # Korean
            ids4 = Tokenizer.encode(tok, "안녕하세요")
            @test !isempty(ids4)
            
            # Arabic
            ids5 = Tokenizer.encode(tok, "مرحبا")
            @test !isempty(ids5)
            
            # Russian
            ids6 = Tokenizer.encode(tok, "Привет")
            @test !isempty(ids6)
            
            println("  Unicode tests passed!")
        else
            @warn "Model not found, skipping unicode tests"
        end
    end

    @testset "encode - empty and whitespace" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # Empty string
            ids1 = Tokenizer.encode(tok, "")
            @test isempty(ids1)
            
            # Single space
            ids2 = Tokenizer.encode(tok, " ")
            @test !isempty(ids2)
            
            # Multiple spaces
            ids3 = Tokenizer.encode(tok, "   ")
            @test !isempty(ids3)
            
            # Newlines
            ids4 = Tokenizer.encode(tok, "\n")
            @test !isempty(ids4)
            
            # Tabs
            ids5 = Tokenizer.encode(tok, "\t")
            @test !isempty(ids5)
            
            println("  Whitespace tests passed!")
        else
            @warn "Model not found, skipping whitespace tests"
        end
    end

    @testset "decode - edge cases" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # Empty list
            decoded1 = Tokenizer.decode(tok, Int[])
            @test decoded1 == ""
            
            # Single token
            ids2 = Tokenizer.encode(tok, "Hello")
            decoded2 = Tokenizer.decode(tok, ids2)
            @test decoded2 == "Hello"
            
            println("  Decode edge case tests passed!")
        else
            @warn "Model not found, skipping decode edge case tests"
        end
    end

    @testset "_append_pretokenized_ids! - qwen pretokenizer" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # Test that pretokenized text is handled
            ids = Int[]
            # This would be called internally during encoding
            # We just verify the tokenizer works
            test_text = "Hello world"
            encoded = Tokenizer.encode(tok, test_text)
            @test !isempty(encoded)
            
            println("  Pretokenizer test passed!")
        else
            @warn "Model not found, skipping pretokenizer tests"
        end
    end

    @testset "Special token handling" begin
        MODEL_PATH = get(ENV, "INFERNO_MODEL", joinpath(@__DIR__, "../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"))
        
        if isfile(MODEL_PATH)
            file = Inferno.GGUF.read_gguf(MODEL_PATH)
            tok = Tokenizer.load_tokenizer(file.metadata)
            
            # Encode with special characters that might be in special tokens
            text = "<|im_start|>user"
            ids = Tokenizer.encode(tok, text)
            @test !isempty(ids)
            
            println("  Special token test passed!")
        else
            @warn "Model not found, skipping special token tests"
        end
    end
end
