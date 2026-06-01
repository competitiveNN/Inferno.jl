#!/usr/bin/env julia
using Inferno
using Inferno.GGUF

file = read_gguf("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf")

# Check tensor shapes in GGUF
for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
             "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
             "blk.0.attn_output.weight", "blk.3.attn_q.weight"]
    if haskey(file.tensors, name)
        tensor = file.tensors[name]
        println("$name: dimensions=$(tensor.dimensions)")
    else
        println("$name: NOT FOUND")
    end
end

# Check SSM tensors
println("\nSSM tensors:")
for name in ["blk.0.ssm_in_proj.weight", "blk.0.ssm_gate_proj.weight", "blk.0.ssm_out.weight",
             "blk.0.ssm_a", "blk.0.ssm_dt_bias"]
    if haskey(file.tensors, name)
        tensor = file.tensors[name]
        println("$name: dimensions=$(tensor.dimensions)")
    else
        println("$name: NOT FOUND")
    end
end
