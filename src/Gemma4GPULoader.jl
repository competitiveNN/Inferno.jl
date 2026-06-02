module Gemma4GPULoader
# Stub: Gemma4 GPU loader code is currently inactive for the CPU path.
# Kept as a placeholder to resolve the `Inferno.jl` include chain.
export load_gemma4_gpu
function load_gemma4_gpu(path::AbstractString)
  throw(ErrorException("Gemma4 GPU loader is not available in this build."))
end
end  # module Gemma4GPULoader

