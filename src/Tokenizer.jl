module Tokenizer

import Base: show
using Unicode
export BPETokenizer, encode, decode, load_tokenizer

"""
Minimal BPE tokenizer that reads the vocabulary from GGUF metadata.
GGUF stores tokens under `tokenizer.ggml.tokens` and merges under
`tokenizer.ggml.merges`.
"""
struct BPETokenizer
    token_to_id::Dict{String, Int}
    id_to_token::Vector{String}
    merges::Vector{Tuple{String, String}}
    merge_priority::Dict{Tuple{String, String}, Int}
    special_tokens::Vector{String}
    pretokenizer::String
    bos_id::Int
    eos_id::Int
end

const QWEN_PRETOKENIZER_RE = Regex(
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

function load_tokenizer(metadata::Dict{String, Any})
    tokens = get(metadata, "tokenizer.ggml.tokens", String[])
    merges_raw = get(metadata, "tokenizer.ggml.merges", String[])
    bos_id_raw = Int(get(metadata, "tokenizer.ggml.bos_token_id", -1))
    eos_id_raw = Int(get(metadata, "tokenizer.ggml.eos_token_id", -1))
    pretokenizer = String(get(metadata, "tokenizer.ggml.pre", "default"))
    token_types = get(metadata, "tokenizer.ggml.token_type", Any[])

    token_to_id = Dict{String, Int}()
    id_to_token = Vector{String}(undef, length(tokens))
    for (i, tok) in enumerate(tokens)
        s = String(tok)
        token_to_id[s] = i  # 1-indexed
        id_to_token[i] = s
    end

    merges = Tuple{String, String}[]
    for m in merges_raw
        parts = split(String(m), ' ', limit=2)
        if length(parts) == 2
            push!(merges, (parts[1], parts[2]))
        end
    end

    merge_priority = Dict{Tuple{String, String}, Int}()
    for (i, pair) in enumerate(merges)
        merge_priority[pair] = i
    end

    special_tokens = String[]
    if token_types isa AbstractVector && length(token_types) == length(tokens)
        for i in eachindex(tokens, token_types)
            tok_type = try
                Int(token_types[i])
            catch
                1
            end
            if tok_type != 1
                tok = String(tokens[i])
                if startswith(tok, "<") && endswith(tok, ">")
                    push!(special_tokens, tok)
                end
            end
        end
    end
    sort!(unique!(special_tokens); by = s -> (-ncodeunits(s), s))

    bos_id = bos_id_raw >= 0 ? bos_id_raw + 1 : bos_id_raw
    eos_id = eos_id_raw >= 0 ? eos_id_raw + 1 : eos_id_raw

    BPETokenizer(token_to_id, id_to_token, merges, merge_priority, special_tokens, pretokenizer, bos_id, eos_id)
end

function get_byte_map()
    bs = vcat(collect(UInt8('!'):UInt8('~')), collect(UInt8('¡'):UInt8('¬')), collect(UInt8('®'):UInt8('ÿ')))
    cs = Int.(bs)
    n = 0
    for b in 0:255
        if !(UInt8(b) in bs)
            push!(bs, UInt8(b))
            push!(cs, 256 + n)
            n += 1
        end
    end
    byte_to_char = Dict{UInt8, Char}()
    for (b, c) in zip(bs, cs)
        byte_to_char[UInt8(b)] = Char(c)
    end
    return byte_to_char
end

const BYTE_TO_CHAR = get_byte_map()
const CHAR_TO_BYTE = Dict(v => k for (k, v) in BYTE_TO_CHAR)

function _append_fallback_ids!(ids::Vector{Int}, tok::BPETokenizer, symbol::String)
    for c in symbol
        b = get(CHAR_TO_BYTE, c, nothing)
        b === nothing && error("Tokenizer fallback failed for symbol: $(repr(symbol))")
        fb = string("<0x", uppercase(string(b, base=16, pad=2)), ">")
        fbid = get(tok.token_to_id, fb, 0)
        fbid > 0 || error("Missing byte fallback token $fb in vocabulary")
        push!(ids, fbid)
    end
end

function _encode_piece!(ids::Vector{Int}, tok::BPETokenizer, piece::AbstractString)
    isempty(piece) && return ids

    symbols = String[]
    for b in codeunits(piece)
        push!(symbols, string(BYTE_TO_CHAR[UInt8(b)]))
    end

    changed = true
    while changed && length(symbols) > 1
        changed = false
        best_idx = 0
        best_pri = typemax(Int)

        @inbounds for i in 1:(length(symbols) - 1)
            pair = (symbols[i], symbols[i + 1])
            pri = get(tok.merge_priority, pair, typemax(Int))
            if pri < best_pri
                best_pri = pri
                best_idx = i
            end
        end

        if best_idx > 0 && best_pri < typemax(Int)
            symbols[best_idx] = symbols[best_idx] * symbols[best_idx + 1]
            deleteat!(symbols, best_idx + 1)
            changed = true
        end
    end

    for s in symbols
        id = get(tok.token_to_id, s, 0)
        if id > 0
            push!(ids, id)
        else
            _append_fallback_ids!(ids, tok, s)
        end
    end

    return ids
end

function _append_pretokenized_ids!(ids::Vector{Int}, tok::BPETokenizer, text::AbstractString)
    isempty(text) && return ids
    normalized = Unicode.normalize(String(text), :NFC)

    if tok.pretokenizer == "qwen2"
        for m in eachmatch(QWEN_PRETOKENIZER_RE, normalized)
            _encode_piece!(ids, tok, m.match)
        end
    else
        _encode_piece!(ids, tok, normalized)
    end

    return ids
end

function _split_special_tokens(tok::BPETokenizer, text::String)
    isempty(tok.special_tokens) && return Any[(false, text)]

    parts = Any[]
    cursor = firstindex(text)

    while cursor <= lastindex(text)
        found_range = nothing
        found_token = nothing

        for special in tok.special_tokens
            r = findnext(special, text, cursor)
            if r !== nothing && (found_range === nothing || first(r) < first(found_range))
                found_range = r
                found_token = special
                first(r) == cursor && break
            end
        end

        if found_range === nothing
            push!(parts, (false, SubString(text, cursor)))
            break
        end

        if first(found_range) > cursor
            push!(parts, (false, SubString(text, cursor, prevind(text, first(found_range)))))
        end

        push!(parts, (true, found_token))
        cursor = nextind(text, last(found_range))
    end

    return parts
end

"""
    encode(tok, text) -> Vector{Int}

Byte-level BPE encoding.
"""
function encode(tok::BPETokenizer, text::String)
    ids = Int[]
    for (is_special, piece) in _split_special_tokens(tok, text)
        if is_special
            id = get(tok.token_to_id, String(piece), 0)
            id > 0 || error("Missing special token in vocabulary: $(repr(piece))")
            push!(ids, id)
        else
            _append_pretokenized_ids!(ids, tok, String(piece))
        end
    end

    return ids
end

"""
    decode(tok, ids) -> String

Decode token IDs back to a UTF-8 string.
"""
function decode(tok::BPETokenizer, ids::Vector{Int})
    output = IOBuffer()
    for id in ids
        if 1 <= id <= length(tok.id_to_token)
            s = tok.id_to_token[id]
            m = match(r"^<0x([0-9A-Fa-f]{2})>$", s)
            if m !== nothing
                write(output, parse(UInt8, m.captures[1], base=16))
            else
                for c in s
                    b = get(CHAR_TO_BYTE, c, nothing)
                    if b !== nothing
                        write(output, b)
                    else
                        write(output, c)
                    end
                end
            end
        end
    end
    return String(take!(output))
end

function show(io::IO, tok::BPETokenizer)
    print(io, "[CPU] BPETokenizer (vocab=$(length(tok.id_to_token)))")
end

end # module
