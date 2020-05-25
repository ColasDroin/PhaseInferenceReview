mutable struct HMM
    l_α::Array{Float64} #forward array
    l_β::Array{Float64} #backward array
    l_γ::Array{Float64} #f-b array
    TRθ::Array{Float64} #transition matrix θ,ϕ->θ
    TRϕ::Array{Float64} #transition matrix θ,ϕ->ϕ
    TRθϕ::Array{Float64} #transition matrix θ,ϕ->θ,ϕ 
                    
    Eθ::Array{Float64} #emission matrix
    Eϕ::Array{Float64} #emission matrix
                                                
    l_cnorm::Array{Float64} #log-likelihood vector
    
    N_θ::Int64
    N_ϕ::Int64
    
    tf::Int64
end


function HMM(;TRθ, TRϕ, domain_phase, Theta, Phi, trace)
    tf = size(trace,1)

    N_θ = size(domain_phase,1)
    N_ϕ = N_θ#size(domain_phase,1)
    
    l_cnorm = Array{Float64}(undef, 0, 2)
    l_α = zeros(tf+1, N_θ, N_ϕ)
    l_β = zeros(tf+1, N_θ, N_ϕ)
    l_γ = zeros(tf+1, N_θ, N_ϕ)

    TRθϕ = ones(N_θ, N_ϕ, N_θ, N_ϕ)
    
    @einsum TRθϕ[a,b,c,d] = TRθ[a,b,c] * TRϕ[a,b,d]
    
    #build emission matrix
    (Eθ, Eϕ) = buildE(domain_phase, Theta, Phi, trace)
    return HMM(l_α, l_β, l_γ, TRθ, TRϕ, TRθϕ, Eθ, Eϕ, l_cnorm, N_θ, N_ϕ, tf)
end
    
function buildE(domain_phase, Theta, Phi, trace)
    #println("start building E")
    resolution = size(domain_phase, 1)
    
    #build emission matrices    
    Eθ = zeros(size(trace,1), resolution)
    Eϕ = zeros(size(trace,1), resolution)

    #prevent divergence
    if Theta.σₑ <= 0
        Theta.σₑ = 2*10^-1
    end
    if Phi.σₑ <= 0
        Phi.σₑ = 2*10^-1    
    end    

    #loop over all states 
    for t in 1:size(trace,1)
        for (idx_theta, θ) in enumerate(domain_phase)                                 
            loc_θ =  ω_fast(Theta, θ)
            Eθ[ t, idx_theta] = exp(-0.5 * ((trace[t,3] -  loc_θ)/ Theta.σₑ   )^2) / ( sqrt(2*π) * Theta.σₑ)
        end
        for (idx_phi, ϕ) in enumerate(domain_phase)                                 
            loc_ϕ =  ω_fast(Phi, ϕ)
            Eϕ[ t, idx_phi] = exp(-0.5 * ((trace[t,4] -  loc_ϕ)/ Phi.σₑ   )^2) / ( sqrt(2*π) * Phi.σₑ)
        end
    end
    return (Eθ, Eϕ)
    println("end building E")
end
    
    
function doForward(hmm::HMM)
    #println("Forward algorithm started")
    α = ones(hmm.N_θ, hmm.N_ϕ)./(hmm.N_θ * hmm.N_ϕ)
    #α = zeros(hmm.N_θ, hmm.N_ϕ)
    #α[1,:] = [1 for x in 1:hmm.N_θ]
    hmm.l_α[1,:,:] = α
    for t in 1:1:hmm.tf
        @tensor begin
        α[c,d] := α[a,b] * hmm.TRθϕ[a,b,c,d]    
        end
        #println("alpha", α)
        
        ##dot product with emission probability
        Eθ = hmm.Eθ[t,:]
        @einsum α[a,b] = α[a,b] * Eθ[a]    
        cnorm1 = sum(α)
        
        α = α / cnorm1

        Eϕ = hmm.Eϕ[t,:]
        @einsum α[a,b] = α[a,b] * Eϕ[b] 
        cnorm2 = sum(α)
        α = α / cnorm2            

        hmm.l_α[t+1,:,:] = α
        hmm.l_cnorm = vcat(hmm.l_cnorm, [cnorm1 cnorm2] )
    end
end


function doBackward(hmm::HMM)
    println("Backward algorithm started")
    β = ones(hmm.N_θ, hmm.N_ϕ)
    hmm.l_α[hmm.tf+1,:,:] = hmm.l_α[hmm.tf+1,:,:] .* β 
    for t in hmm.tf:-1:1

        #dot product with emission probability
        Eθ = hmm.Eθ[t,:]
        @einsum β[a,b] = β[a,b] * Eθ[a]
        β = β ./ hmm.l_cnorm[t,1]

        

        Eϕ = hmm.Eϕ[t,:]
        @einsum β[a,b] = β[a,b] * Eϕ[b] 
        β = β ./ hmm.l_cnorm[t,2]  
                 
        #compute transition probabilities
        @tensor begin
        β[a,b] := β[c,d] * hmm.TRθϕ[a,b,c,d]    
        end
          
        hmm.l_α[t,:,:] = hmm.l_α[t,:,:] .* β
        
    end
end

function logP(hmm::HMM)
    l_logP_1 = hmm.l_cnorm[:,1]
    l_logP_2 = hmm.l_cnorm[:,2]
    return (sum(log.(l_logP_1))/length(l_logP_1),sum(log.(l_logP_2))/length(l_logP_2))
end

function forwardBackward(hmm::HMM)
    doForward(hmm)
    #doBackward(hmm)        
    (ll1, ll2) = logP(hmm)
    return (ll1, ll2)
end
