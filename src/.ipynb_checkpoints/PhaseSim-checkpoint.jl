mutable struct PhaseSim
    Θ::Phase
    Φ::Phase
    dt::Float64
end

function PhaseSim(Θ, Φ, dt)
    #nothing is done for now
    return PhaseSim(Θ, Φ, dt)
end

function iterate_process(phase_sim::PhaseSim)
    
    #compute frequency
    ω_θ_inst = phase_sim.Θ.f + F_fast(phase_sim.Θ, phase_sim.Θ.ϕ,phase_sim.Φ.ϕ)
    ω_ϕ_inst = phase_sim.Φ.f + F_fast(phase_sim.Φ, phase_sim.Θ.ϕ,phase_sim.Φ.ϕ)
    
    #compute phase
    phase_sim.Θ.ϕ  = phase_sim.Θ.ϕ  + ω_θ_inst*phase_sim.dt  + phase_sim.Θ.σ_ϕ*randn()*dt
    phase_sim.Φ.ϕ  = phase_sim.Φ.ϕ  + ω_ϕ_inst*phase_sim.dt  + phase_sim.Φ.σ_ϕ*randn()*dt
    
    #compute amplitude
    phase_sim.Θ.A  = phase_sim.Θ.A -phase_sim.Θ.γₐ*(phase_sim.Θ.A - phase_sim.Θ.μₐ)*phase_sim.dt  + phase_sim.Θ.σₐ*randn()*dt
    phase_sim.Φ.A  = phase_sim.Φ.A -phase_sim.Φ.γₐ*(phase_sim.Φ.A - phase_sim.Θ.μₐ)*phase_sim.dt  + phase_sim.Φ.σₐ*randn()*dt

    #compute background
    phase_sim.Θ.B  = phase_sim.Θ.B -phase_sim.Θ.γᵦ*(phase_sim.Θ.B - phase_sim.Θ.μᵦ)*phase_sim.dt  + phase_sim.Θ.σᵦ*randn()*dt
    phase_sim.Φ.B  = phase_sim.Φ.B -phase_sim.Φ.γᵦ*(phase_sim.Φ.B - phase_sim.Θ.μᵦ)*phase_sim.dt  + phase_sim.Φ.σᵦ*randn()*dt
    
    signal_θ = phase_sim.Θ.A*ω_fast(phase_sim.Θ)+phase_sim.Θ.B+phase_sim.Θ.σₑ*randn()
    signal_ϕ = phase_sim.Φ.A*ω_fast(phase_sim.Φ)+phase_sim.Φ.B+phase_sim.Φ.σₑ*randn()
    return [ phase_sim.Θ.ϕ phase_sim.Φ.ϕ signal_θ signal_ϕ ]
end