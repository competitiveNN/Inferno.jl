"""
Interactive chat module for Inferno.jl.

Provides a REPL-like chat interface with conversation history management.
"""
module Chat

using Base.Terminals
using Base: join
using ..Inferno: stream_to_stdout, generate_cpu, generate_stream_cpu
using ..Inferno.Tokenizer: encode, decode

export chat!, start_chat, Message, build_prompt

struct Message
    role::String
    content::String
end

Message(role::Symbol, content) = Message(String(role), String(content))

function build_prompt(messages::Vector{Message}; enable_thinking::Bool=false)
 parts = String[]
 for msg in messages
 if msg.role == "system"
 push!(parts, "<|im_start|>system\n$(msg.content)<|im_end|>")
 elseif msg.role == "user"
 push!(parts, "<|im_start|>user\n$(msg.content)<|im_end|>")
 elseif msg.role == "assistant"
 push!(parts, "<|im_start|>assistant\n$(msg.content)<|im_end|>")
 end
 end
 # Qwen3.5 chat template: generation prompt includes <think> tokens
 # Default (enable_thinking=false): skip thinking, go straight to answer
 #   <think>\n\n</think>\n\n  (empty think block tells model to answer directly)
 # With thinking: model will produce chain-of-thought in <think> block
 #   <think>\n  (open think block, model continues thinking then closes with </think>)
 if enable_thinking
 push!(parts, "<|im_start|>assistant\n<think>\n")
 else
 push!(parts, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
 end
 return join(parts, "\n")
end

function render_with_thinking_color(response::String, term)
    # No thinking blocks - print normally
    if !occursin("<think>", response)
        print(term, response)
        println(term)
        return response
    end
    
    # Split response into parts around think blocks
    parts = split(response, "<think>")
    buffer = IOBuffer()
    
    for (i, part) in enumerate(parts)
        if i == 1
            # Before first think block - print normally
            write(buffer, part)
        else
            # Inside think block - find where it ends
            if occursin("</think>", part)
                subparts = split(part, "</think>")
                think_content = subparts[1]
                remaining = join(subparts[2:end], "</think>")
                
                # Print think block in gray
                printstyled(buffer, "<think>", color=:light_black)
                printstyled(buffer, think_content, color=:light_black)
                printstyled(buffer, "</think>", color=:light_black)
                
                # Print remaining normally
                write(buffer, remaining)
            else
                # Unclosed think block - print in gray
                printstyled(buffer, "<think>", color=:light_black)
                printstyled(buffer, part, color=:light_black)
            end
        end
    end
    
    print(term, String(take!(buffer)))
    println(term)
    flush(term)
    return response
end

function stream_with_colors(model, tok, prompt; io::IO=stdout, stop_tokens::Set{Int}=Set{Int}(), max_tokens::Int=512, thinking_enabled::Bool=false, show_tps::Bool=false, kwargs...)
    # Stream generation with thinking block color tracking
    prompt_tokens = encode(tok, prompt)
    is_thinking = thinking_enabled
    token_buffer = ""
    
    # Track tokens and time for TPS
    t0 = time()
    token_count = 0
    
    # Extract and convert Float32 parameters
    temperature = haskey(kwargs, :temperature) ? Float32(kwargs[:temperature]) : 0.7f0
    top_p = haskey(kwargs, :top_p) ? Float32(kwargs[:top_p]) : 0.95f0
    top_k = haskey(kwargs, :top_k) ? kwargs[:top_k] : 20
    repetition_penalty = haskey(kwargs, :repetition_penalty) ? Float32(kwargs[:repetition_penalty]) : 1.0f0
    presence_penalty = haskey(kwargs, :presence_penalty) ? Float32(kwargs[:presence_penalty]) : 0.0f0
    min_p = haskey(kwargs, :min_p) ? Float32(kwargs[:min_p]) : 0.0f0
    
    for token in generate_stream_cpu(model, prompt_tokens, (ids) -> decode(tok, ids); 
        max_tokens=max_tokens, stop_tokens=stop_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, min_p=min_p,
        show_tps=false, io=stdout)
        token_buffer *= token
        token_count += 1
        
        # Print with appropriate color - color content in thinking blocks
        if is_thinking
            printstyled(io, token, color=:light_black, italic=true)
        else
            print(io, token)
        end
flush(io)
    end
    
    # Print TPS if enabled
    if show_tps && token_count > 0
        elapsed = time() - t0
        tps = elapsed > 0 ? token_count / elapsed : 0.0
        printstyled(io, "\n[t/s] $(round(tps, digits=2)) tokens/s — $(token_count) tokens in $(round(elapsed, digits=3))s\n", color=:cyan)
    end
    
    return token_buffer
end

function chat(model, tok, messages::Vector{Message}; enable_thinking::Bool=false, kwargs...)
 prompt = build_prompt(messages; enable_thinking=enable_thinking)
 # Stop on EOS and <|im_end|> tokens
 im_end_id = get(tok.token_to_id, "<|im_end|>", 0)
 stop_token = im_end_id != 0 ? im_end_id : tok.eos_id
 return stream_to_stdout(model, tok, prompt; stop_token=stop_token, kwargs...)
end

const interrupt_flag = Threads.Atomic{Bool}(false)
const chat_terminal = Ref{Any}(nothing)

function check_interrupt()
    if interrupt_flag[]
        interrupt_flag[], false
        return true
    end
    # Try to check if there's input available on terminal
    if chat_terminal[] !== nothing
        try
            t = chat_terminal[]
            if bytesavailable(t) > 0
                c = read(t, Char)
                if c == '\x03'
                    return true
                end
            end
        catch e
            # Ignore errors - might not be available
        end
    end
    # Also try stdin directly (might work in some cases)
    try
        if bytesavailable(stdin) > 0
            c = read(stdin, Char)
            if c == '\x03'
                return true
            end
        end
    catch e
        # Ignore
    end
    return false
end

mutable struct ChatState
    history::Vector{String}
    hist_pos::Int
end

function iswordchar(c::Char)
    c != ' ' && c != '\t'
end

function backward_word(cursor::Int, buffer::Vector{Char})
    cursor <= 1 && return 1
    len = length(buffer)
    pos = min(cursor - 1, len)
    pos <= 0 && return 1
    
    if iswordchar(buffer[pos])
        while pos > 1 && iswordchar(buffer[pos-1])
            pos -= 1
        end
        return pos
    end
    
    while pos >= 1 && !iswordchar(buffer[pos])
        pos -= 1
    end
    while pos >= 1 && iswordchar(buffer[pos])
        pos -= 1
    end
    return max(1, pos + 1)
end

function forward_word(cursor::Int, buffer::Vector{Char})
    len = length(buffer)
    cursor > len && return len + 1
    pos = cursor
    while pos <= len && iswordchar(buffer[pos])
        pos += 1
    end
    while pos <= len && !iswordchar(buffer[pos])
        pos += 1
    end
    return pos
end

function clear_line(term, prompt)
    print(term, "\r" * " "^80 * "\r" * prompt)
end

function refresh_line(term, prompt, buffer, cursor)
    print(term, "\r" * " "^80 * "\r" * prompt * String(buffer) * "\r")
    print(term, "\r" * prompt)
    for i in 1:cursor-1
        print(term, buffer[i])
    end
end

function read_line_chat(term, state)
    print(term, "You> ")
    flush(term)
    
    buffer = Char[]
    cursor = 1
    state.hist_pos = -1
    
    while true
        c = try
            read(term, Char)
        catch e
            if e isa EOFError
                return "EXIT_CHAT"
            else
                rethrow(e)
            end
        end
        
        # Skip escape sequences ( CSI, OSC, DCS, etc. )
        if c == '\e'
            # This is start of escape sequence - consume and discard
            continue
        end
        
        if c == '\x04'
            isempty(buffer) && return "EXIT_CHAT"
            println(term)
            result = String(buffer)
            !isempty(result) && push!(state.history, result)
            return result
        elseif c == '\x03'
            println(term)
            return ""
        elseif c == '\x10'  # Ctrl+P = previous history
            if !isempty(state.history)
                if state.hist_pos < length(state.history) - 1
                    state.hist_pos += 1
                elseif state.hist_pos == -1
                    state.hist_pos = 0
                end
                buffer = collect(state.history[end - state.hist_pos])
                cursor = length(buffer) + 1
                refresh_line(term, "You> ", buffer, cursor)
            end
        elseif c == '\x0e'  # Ctrl+N = next history
            if state.hist_pos > 0
                state.hist_pos -= 1
            elseif state.hist_pos == 0
                state.hist_pos = -1
            end
            if state.hist_pos >= 0 && state.hist_pos < length(state.history)
                buffer = collect(state.history[end - state.hist_pos])
            else
                buffer = Char[]
            end
            cursor = length(buffer) + 1
            refresh_line(term, "You> ", buffer, cursor)
        elseif c == '\r' || c == '\n'
            # Check if this might be part of a paste operation
            # Try to read more characters - if we get multiple quickly, it's paste
            potential_paste = Char[]
            try
                # Non-blocking check for more input
                start_time = time()
                while time() - start_time < 0.01  # 10ms window
                    if bytesavailable(term) > 0
                        push!(potential_paste, read(term, Char))
                    else
                        sleep(0.001)
                    end
                end
            catch
            end
            
            if !isempty(potential_paste)
                # This appears to be paste - include in buffer but don't render
                paste_len = length(potential_paste)
                truncate_msg = paste_len > 100 ? "[$paste_len chars truncated]" : "[$paste_len chars pasted]"
                println(term)
                printstyled(truncate_msg, color=:cyan)
                print(term, "\r\nYou> ")
                flush(term)
                # Add to buffer but don't print each char
                for pc in potential_paste
                    push!(buffer, pc)
                end
                # Continue editing - reset state and continue the input loop
                cursor = length(buffer) + 1
                refresh_line(term, "You> ", buffer, cursor)
            else
                # Normal enter - submit
                println(term)
                result = String(buffer)
                !isempty(result) && push!(state.history, result)
                return result
            end
        elseif c == '\e'
            next = read(term, Char)
            if next == '['
                seq = read(term, Char)
                if seq == 'A' && !isempty(state.history)
                    if state.hist_pos < length(state.history) - 1
                        state.hist_pos += 1
                    elseif state.hist_pos == -1
                        state.hist_pos = 0
                    end
                    buffer = collect(state.history[end - state.hist_pos])
                    cursor = length(buffer) + 1
                    refresh_line(term, "You> ", buffer, cursor)
                elseif seq == 'B'
                    if state.hist_pos > 0
                        state.hist_pos -= 1
                    elseif state.hist_pos == 0
                        state.hist_pos = -1
                    end
                    if state.hist_pos >= 0 && state.hist_pos < length(state.history)
                        buffer = collect(state.history[end - state.hist_pos])
                    else
                        buffer = Char[]
                    end
                    cursor = length(buffer) + 1
                    refresh_line(term, "You> ", buffer, cursor)
                elseif seq == 'C'
                    if cursor <= length(buffer)
                        print(term, buffer[cursor])
                        cursor += 1
                    end
                elseif seq == 'D'
                    if cursor > 1
                        cursor -= 1
                        print(term, "\b")
                    end
                end
            elseif next == '\x7f'
                new_cursor = backward_word(cursor, buffer)
                if new_cursor < cursor
                    deleteat!(buffer, new_cursor:cursor-1)
                    cursor = new_cursor
                    refresh_line(term, "You> ", buffer, cursor)
                end
            elseif next == 'b'
                cursor = backward_word(cursor, buffer)
                refresh_line(term, "You> ", buffer, cursor)
            elseif next == 'f'
                cursor = forward_word(cursor, buffer)
                refresh_line(term, "You> ", buffer, cursor)
            elseif next == 'd'
                word_end = forward_word(cursor, buffer)
                if word_end > cursor
                    deleteat!(buffer, cursor:word_end-1)
                    refresh_line(term, "You> ", buffer, cursor)
                end
            end
            continue
        elseif c == '\x01'
            cursor = 1
            refresh_line(term, "You> ", buffer, cursor)
            continue
        elseif c == '\x05'
            for i in cursor:length(buffer)
                print(term, buffer[i])
            end
            cursor = min(length(buffer) + 1, cursor)
            continue
        elseif c == '\x02'
            if cursor > 1
                cursor -= 1
                print(term, "\b")
            end
            continue
        elseif c == '\x06'
            if cursor <= length(buffer)
                print(term, buffer[cursor])
                cursor += 1
            end
            continue
        elseif c == '\x0b'
            buffer = buffer[1:cursor-1]
            refresh_line(term, "You> ", buffer, cursor)
            continue
        elseif c == '\x15'
            buffer = buffer[cursor:end]
            cursor = 1
            refresh_line(term, "You> ", buffer, cursor)
            continue
        elseif c == '\x7f'
            if cursor > 1
                deleteat!(buffer, cursor - 1)
                cursor -= 1
                refresh_line(term, "You> ", buffer, cursor)
            end
            continue
        else
            if cursor <= length(buffer)
                insert!(buffer, cursor, c)
                refresh_line(term, "You> ", buffer, cursor)
            else
                push!(buffer, c)
                print(term, c)
            end
            cursor += 1
        end
    end
end

function chat!(model, tok; system_prompt::String="You are a helpful assistant.", enable_thinking::Bool=false, max_tokens::Int=512, temperature::Float32=1.0f0, top_p::Float32=0.95f0, top_k::Int=20, repetition_penalty::Float32=1.0f0, presence_penalty::Float32=0.0f0, min_p::Float32=0.0f0, show_tps::Bool=true, kwargs...)
 messages = [Message(:system, system_prompt)]
 thinking_mode = enable_thinking
 
 banner = """
 ╔═══════════════════════════════════════════════════╗
 ║ Welcome to Inferno Chat! ║
 ╠═══════════════════════════════════════════════════╣
 ║ Type your message and press Enter to chat. ║
 ║ Commands: ║
 ║ /clear - Clear conversation history ║
 ║ /system - Change system prompt ║
 ║ /think - Toggle thinking mode ║
 ║ /quit - Exit chat ║
 ╚═══════════════════════════════════════════════════╝
 """
 
 printstyled(banner, color=:cyan, bold=true)
 printstyled("Thinking: $(thinking_mode ? "ON" : "OFF")\n", color=:yellow)
 flush(stdout)
 
 state = ChatState(String[], -1)
 term = TTYTerminal("/dev/tty", stdin, stdout, stderr)
 chat_terminal[], term
 interrupt_flag[], false
 
# Stop on EOS and <|im_end|>
  im_end_id = get(tok.token_to_id, "<|im_end|>", 0)
  stop_token = im_end_id != 0 ? im_end_id : tok.eos_id
 
 try
 raw!(term, true)
 
 while true
 line = read_line_chat(term, state)
 
 isempty(line) && continue
 line == "EXIT_CHAT" && (printstyled("Goodbye!\n", color=:cyan); break)
 line == "/quit" || line == "/exit" && (printstyled("Goodbye!\n", color=:cyan); break)
 line == "/clear" && (messages = [Message(:system, system_prompt)]; printstyled("Conversation cleared.\n", color=:yellow); continue)
 line == "/think" && (thinking_mode = !thinking_mode; printstyled("Thinking: $(thinking_mode ? "ON" : "OFF")\n", color=:yellow); continue)
 line == "/system" && begin
 printstyled("New system prompt: ", color=:magenta)
 raw!(term, false)
 new_system = readline(stdin)
 raw!(term, true)
 !isempty(new_system) && (system_prompt = new_system; messages = [Message(:system, system_prompt)]; printstyled("System prompt updated!\n", color=:green))
 continue
 end
 startswith(line, "/") && (printstyled("Unknown command: $(line)\n", color=:red); continue)
 
 push!(state.history, line)
 push!(messages, Message(:user, line))
 prompt = build_prompt(messages; enable_thinking=thinking_mode)
 
# Exit raw mode during generation - makes stdin line-buffered
  raw!(term, false)
  
  # Generate and stream with thinking colors
  im_end_id = get(tok.token_to_id, "<|im_end|>", 0)
  stop_tokens = Set(filter(!=(0), [tok.eos_id, im_end_id]))
response = stream_with_colors(model, tok, prompt; stop_tokens=stop_tokens, max_tokens=max_tokens, io=term, thinking_enabled=thinking_mode, show_tps=show_tps, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty, min_p=min_p, kwargs...)
   
   # Print newline after response
   println(term)
   
   # Re-enter raw mode for input
   raw!(term, true)
 
 push!(messages, Message(:assistant, response))
 end
    catch e
        if e isa InterruptException
            println(term)
            interrupt_flag[], false
        else
            rethrow(e)
        end
    finally
        raw!(term, false)
    end
end

function start_chat(model, tok; kwargs...)
    chat!(model, tok; kwargs...)
end

end