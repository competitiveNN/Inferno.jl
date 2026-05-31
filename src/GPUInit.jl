"""
    GPUInit.jl - Level Zero initialization for oneAPI.jl

Now that Inferno uses upstream oneAPI (not vendored), the bundled NEO_jll
at v25.44.36015 provides its own Level Zero library via Julia Artifacts.
No pre-init is needed — oneAPI.jl handles its own init.

The system intel-compute-runtime (v26.18.38308.1) crashes on Battlemage
B580 (Xe2) with a GMM abort, so we must NOT call system zeInit.

GPU compute remains blocked: NEO_jll v25.44 is too old for Xe2 ISA,
and the system driver v26.18 crashes. A reboot after intel-gpu-firmware
update may fix the system driver, or a newer NEO_jll may ship.
"""
module GPUInit

const _initialized = Ref{Bool}(false)

function __init__()
    # This module is intentionally a no-op since:
    # 1. Upstream oneAPI.jl uses its own bundled NEO_jll artifacts
    # 2. System Level Zero driver crashes on zeInit (GMM abort)
    # 3. NEO_jll v25.44 is too old for Battlemage Xe2 ISA
    _initialized[] = true
    @info "GPUInit: oneAPI.jl handles its own init. See src/GPUInit.jl for GPU driver status."
end

"""Check if GPU module is ready (always true — oneAPI handles itself)"""
functional() = true

end # module GPUInit