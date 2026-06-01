using Inferno
using Statistics

function main()
    # Test with just embedding and lm_head
    cpu_model, file = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Test the embedding -> lm_head path
    # Get embedding for "The" (token 760)
    emb = cpu_model.embed[:, 760]
    println("Embedding norm: ", sqrt(sum(abs2, emb)))

    # Apply lm_head directly (without model layers)
    logits_direct = cpu_model.lm_head * emb
    println("\nDirect logits (emb -> lm_head):")
    println("  mean: ", mean(logits_direct))
    println("  std: ", std(logits_direct))
    println("  max: ", maximum(logits_direct))
    println("  min: ", minimum(logits_direct))

    # Top 10
    top10 = sortperm(logits_direct, rev=true)[1:10]
    tokens = file.metadata["tokenizer.ggml.tokens"]
    println("\n  Top 10:")
    for t in top10
        println("    Token $t: ", repr(tokens[t+1]), " (logit=", logits_direct[t], ")")
    end
end

main()
