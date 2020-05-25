mutable struct Phase
    name_ϕ::String
    T::Float64
    f::Float64
    σ_ϕ::Float64
    ϕ₀::Float64
    ϕ::Float64
    A::Float64
    B::Float64
    σₑ::Float64
    
    #waveform
    FT_ω::Vector{Complex{Float64}}
    n_harm_ω::Int64
    res_ω::Int64
    #fast waveform
    fast_ω
    #coupling
    FT_F::Matrix{Complex{Float64}}
    d1::Int64
    d2::Int64
    n_harm_F::Int64
    #fast coupling
    fast_F_sim 
    fast_F_inf
    
    #O-U processes parameters
    μₐ::Float64
    γₐ::Float64
    σₐ::Float64
    μᵦ::Float64
    γᵦ::Float64
    σᵦ::Float64
end
    
function Phase(name_ϕ; T=24., σ_ϕ = 0.15, FT_ω = nothing, σₑ = 0.15, FT_F = nothing, ϕ₀ = nothing, μₐ=0.,γₐ = 0., σₐ = 0., μᵦ = 0., γᵦ = 0., σᵦ = 0.)
    f = 2*π/T
    ϕ₀ = ϕ₀!=nothing ? ϕ₀ : rand()*2*π
    ϕ = ϕ₀
    A = 1.
    B = 0.
    #waveform
    n_harm_ω = size(FT_ω,1)
    res_ω = size(FT_ω,1)
    #fast waveform
    fast_ω = nothing
    #coupling
    FT_coupling = FT_F
    d1 = size(FT_F,1)
    d2 = size(FT_F,2)
    n_harm_F = size(FT_F,1)
    #fast coupling
    fast_F_sim = nothing
    fast_F_inf = nothing
    return Phase(name_ϕ, T, f, σ_ϕ, ϕ₀, ϕ, A, B, σₑ, FT_ω, n_harm_ω, res_ω, fast_ω, FT_F, d1, d2, n_harm_F, fast_F_sim, fast_F_inf, μₐ,γₐ, σₐ, μᵦ, γᵦ, σᵦ)
end    

function ω(phase::Phase, ϕ = nothing)
    if ϕ == nothing 
        ϕ = phase.ϕ
    end
    return 1/phase.res_ω*sum(  [  real(phase.FT_ω[k])*cos( (k-1)*ϕ)-imag( phase.FT_ω[k]*sin( (k-1)*ϕ) ) for k in range(1,phase.n_harm_ω) ]   )
end

function ω_fast(phase::Phase, ϕ = nothing)
    if ϕ==nothing
        ϕ = phase.ϕ
    end
    if phase.fast_ω==nothing
        ω = real.(ifft(phase.FT_ω))
        ω_temp = vcat(ω,ω,ω)
        #itp = interpolate(ω_temp, BSpline(Linear()))
        #phase.fast_ω = Interpolations.scale(itp, range(-2*π, stop=4*π, length=length(ω_temp)+1)[1:end-1] )
        dom = collect(range(-2*π, stop=4*π, length=length(ω_temp)+1)[1:end-1])
        knots = (dom,)
        phase.fast_ω = interpolate(knots, ω_temp, Gridded(Linear()))
        
    end
    if length(ϕ)==1
        return phase.fast_ω(mod2pi(ϕ))
    else
        return [phase.fast_ω(mod2pi(x)) for x in ϕ]
    end
end


function F(phase::Phase, ϕ₁,ϕ₂, FT = nothing, n_harm = nothing)
    if FT!=nothing
        FT_F = FT
        d1 = size(FT,1)
        d2 = size(FT,2)
        if n_harm!=nothing
            n_harm_F = n_harm
        else
            n_harm_F = size(FT,1)
            println("n_harm set to maximum since not specified")
        end
    else
        FT_F = phase.FT_F
        d1 = phase.d1
        d2 = phase.d2
        n_harm_F = phase.n_harm_F
    end

    return 1/(d1*d2)*sum([(real(FT_F[k,l])*cos( (k-1)*ϕ₁+(l-1)*ϕ₂) -imag(FT_F[k,l]*sin( (k-1)*ϕ₁+(l-1)*ϕ₂))) for k in range(1,n_harm_F), l in range(1,n_harm_F)] )
end
        
        
function F_fast(phase::Phase, ϕ₁,ϕ₂, FT = nothing)
    #show(FT)
    if FT==nothing
        if phase.fast_F_sim==nothing
            F = real.(ifft(phase.FT_F))
            F_temp = hcat(F,F,F)
            F_temp = vcat(F_temp, F_temp, F_temp)
            itp = interpolate(F_temp, BSpline(Linear()))
            phase.fast_F_sim = Interpolations.scale(itp, range(-2*π, stop=4*π, length=3*size(F,1)+1)[1:end-1], range(-2*π, stop=4*π, length=3*size(F,2)+1)[1:end-1] )
        end
        return phase.fast_F_sim(mod2pi(ϕ₁),mod2pi(ϕ₂))
    else
        #println("OK")
        if phase.fast_F_inf==nothing
            #println("OK2")                
            F = real.(ifft(FT))
                            
            #println(F)    
            #plot_F(F)
                            
            F_temp = hcat(F,F,F)
            F_temp = vcat(F_temp, F_temp, F_temp)
                        
            itp = interpolate(F_temp, BSpline(Linear()))
            phase.fast_F_inf = Interpolations.scale(itp, range(-2*π, stop=4*π, length=3*size(F,1)+1)[1:end-1], range(-2*π, stop=4*π, length=3*size(F,2)+1)[1:end-1] )
        end
        return phase.fast_F_inf(mod2pi(ϕ₁),mod2pi(ϕ₂))
    end
end

function reset_ϕ(phase::Phase; random = true)
    if !random
        phase.ϕ = phase.ϕ₀
    else
        phase.ϕ = rand()*2*π
    end
end

function reset_F(phase::Phase)
    phase.fast_F_sim = nothing
    phase.fast_F_inf = nothing
end  