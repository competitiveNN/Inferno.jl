using Inferno
using Statistics

function main()
    model, _ = load_model_cpu("test/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

    # Create a test vector
    test = randn(Float32, 1024) * 0.1f0
    println("Test input norm: ", sqrt(sum(abs2, test)))

    # Apply final norm
    output = model.final_norm(test)
    println("After final_norm: norm=", sqrt(sum(abs2, output)))
    println("  weight sum: ", sum(model.final_norm.weight))
    println("  weight mean: ", mean(model.final_norm.weight))
    println("  weight std: ", std(model.final_norm.weight))

    # Manual computation
    ss = sum(abs2, test)
    m = ss / length(test)
    scale = 1.0f0 / sqrt(m + model.final_norm.eps)
    expected = test .* scale .* model.final_norm.weight
    println("\nManual computation:")
    println("  ss: ", ss)
    println("  m: ", m)
    println("  scale: ", scale)
    println("  output norm: ", sqrt(sum(abs2, expected)))
end

main()
