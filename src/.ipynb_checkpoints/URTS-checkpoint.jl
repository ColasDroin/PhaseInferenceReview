CIRCULAR = false
mutable struct URTS
    x::Vector{Float64} #state estimate vector
    P::Matrix{Float64} #covariance estimate matrix
    R::Matrix{Float64} #measurement noise matrix
    Q::Matrix{Float64} #process noise matrix
    K::Matrix{Float64} #Kalman gain
    y::Vector{Float64} #innovation residual
    z::Vector{Float64} #measurment
    S::Matrix{Float64} # system uncertainty
    SI::Matrix{Float64} # inverse system uncertainty
    
    dim_x::Int32
    dim_z::Int32
    num_sigmas::Int32
    dt::Float64
    fx 
    hx
    
    # weights for the means and covariances.
    Wm::Vector{Float64}
    Wc::Vector{Float64}

    sigmas_f::Matrix{Float64}
    sigmas_h::Matrix{Float64}
    
    α::Float64
    β::Float64
    κ::Float64
    
    zp::Vector{Float64}
    
    ll::Vector{Float64} #log-likelihood vector
end

"""
URTS constructor
"""
function URTS(;dim_x, dim_z, dt, hx, fx, α, β, κ)
    x = zeros(dim_x)
    P = Matrix(I, dim_x, dim_x)
    Q = Matrix(I, dim_x, dim_x)
    R = Matrix(I, dim_x, dim_x)
    dim_x = dim_x
    dim_z = dim_z
    num_sigmas = 2*dim_x + 1

    # weights for the means and covariances.
    Wc, Wm = compute_weights(dim_x, α, β, κ)

    sigmas_f = zeros(num_sigmas, dim_x)
    sigmas_h = zeros(num_sigmas, dim_z)

    K = zeros(dim_x, dim_z)    # Kalman gain
    y = zeros(dim_z)           # residual
    
    z = zeros(dim_z )  # measurement
    S = zeros(dim_z, dim_z)    # system uncertainty
    SI = zeros(dim_z, dim_z)   # inverse system uncertainty
    
    zp = zeros(dim_z)
    ll = Float64[]
    
    return URTS(x, P, R, Q, K, y, z, S, SI, dim_x, dim_z, num_sigmas, dt, fx, hx, Wm, Wc, sigmas_f, sigmas_h, α, β, κ, zp, ll)
end
    
# TO BE CORRECTED FOR ANGULAR VARIABLE
function unscented_transform(urts::URTS, pass = "state")
    if pass == "state"
        
        if CIRCULAR
            (kmax, n) = size(urts.sigmas_f)

            #x = sigma_angular_mean(urts.sigmas_f, urts.Wm)
            x_comp = exp.(1im.*urts.sigmas_f') * urts.Wm
            x = mod2pi.(angle.(x_comp))
            #println("mod x ", abs.(x_comp))

            #easy fix if phase ill-defined
            x_linear = urts.sigmas_f' * urts.Wm
            diff = angular_subtract(x_linear, x)
            if abs(diff[1])>0.05  #abs(x_comp[1])<0.25
                x[1] = x_linear[1] #mean(urts.sigmas_f[:,1])
                #println("CORRECTION")
                #println(x_linear)
            end
            if abs(diff[2])>0.05#abs(x_comp[2])<0.25
                x[2] = x_linear[2]#mean(urts.sigmas_f[:,2])
                #println("CORRECTION2")
            end    

            #println("urts.x", urts.x)
            #println("x1 ", x)

            P = zeros(n, n)
            #y = exp.(1im * (urts.sigmas_f .- x'))
            #P = angle.(y' * (urts.Wc .* y))

            for k in 1:kmax
            #    
                #y = exp.(1im*(urts.sigmas_f[k,:] .- x))
                y = angular_subtract(urts.sigmas_f[k,:], x)
                #println("sigmas_f", urts.sigmas_f[k,:])
                #println("x", x)
                #println("ytemp ", y)
                #println(angle.((y.*y')))
                P += urts.Wc[k] .* (y.*y')
            end

            P += urts.Q

            #println("URTS.P", urts.P)
            #println("P1 ", P)
            #if sum(abs.(P))>10
            #    println("ALERTTTTTTTTTTTTTTTTTTTTT")
            #end

            #fix in case of divergence
            #if (P[1,1]<0 || P[2,2]<0)
            #    println("DIVERGENCE")
            #    println("P", P)
            #    println("URTS.P", urts.P)
            #    P[1,1] = 1
            #    P[2,2] = 1
            #    P[1,2] = 10^(-5)
            #    P[2,1] = 10^(-5)
            #end

            #if (P[1,1]>1.5 || P[2,2]>1.5)
            #    println("DIVERGENCE")
            #    println("P", P)
            #    println("URTS.P", urts.P)
            #    P[1,1] = 1
            #    P[1,2] = 10^(-5)
            #    P[2,2] = 1
            #    P[2,1] = 10^(-5)
            #end

            #fix in case of divergence
            #if (P[1,1]<0 || P[2,2]<0)
            #    println("DIVERGENCE")
            #    x = urts.sigmas_f' * urts.Wm
            #    y = urts.sigmas_f .- x'
            #    P = y' * (urts.Wc .* y)
            #    P += urts.Q
            #end

            urts.x = x
            urts.P = P
        else
        
            #x = mod2pi.(angle.(exp.(1im.*urts.sigmas_f') * urts.Wm))
            x = urts.sigmas_f' * urts.Wm
            y = urts.sigmas_f .- x'


            #println(size(y))
            #println(size(urts.Wc))
            #println(size(urts.Wc .* y))

            P = y' * (urts.Wc .* y)
            #println("x1 ", x)
            #println("P1 ", P)

            P += urts.Q

            urts.x = x
            urts.P = P
        end
        
        
    elseif pass == "smooth"
        
        if CIRCULAR
            (kmax, n) = size(urts.sigmas_f)

            #x = sigma_angular_mean(urts.sigmas_f, urts.Wm)
            x_comp = exp.(1im.*urts.sigmas_f') * urts.Wm
            x = mod2pi.(angle.(x_comp))

            P = zeros(n, n)

            for k in 1:kmax
                y = angular_subtract(urts.sigmas_f[k,:], x)
                P += urts.Wc[k] .* (y.*y')
            end

            P += urts.Q

            urts.x = x
            urts.P = P
        else
                        
            x = urts.sigmas_f' * urts.Wm
            y = urts.sigmas_f .- x'

            P = y' * (urts.Wc .* y)

            P += urts.Q

            urts.x = x
            urts.P = P
        end
                        
            
            
    elseif pass == "obs"
    
        x = urts.sigmas_h' * urts.Wm
        y = urts.sigmas_h .- x'
        P = y' * (urts.Wc .* y)
        P += urts.R

        urts.zp = x
        urts.S = P
    end
            
end                            
                            
                            
function predict(urts::URTS; l_arg_f = [])
    sigmas = sigma_points(urts.α, urts.β, urts.κ, urts.x, urts.P)
    
    for i in 1:size(sigmas,1)
        urts.sigmas_f[i,:] = urts.fx(sigmas[i,:], urts.dt, l_arg_f...)
    end
    
    #println(urts.sigmas_f[:,:])
        
    #println("first print predict", urts.x, urts.P, urts.sigmas_f, urts.Wm, urts.Wc, urts.Q)
        
    unscented_transform(urts, "state")
    #println("second print predict", urts.x, urts.P)
end


function update(urts::URTS, z; l_arg_h = [])
    #println("WTF")
    for i in 1:size(urts.sigmas_f,1)
        urts.sigmas_h[i,:] =  urts.hx(urts.sigmas_f[i,:], l_arg_h...)
    end
    
    #println("first print update", urts.x, urts.P, urts.sigmas_h, z, urts.R)
    
    # mean and covariance of prediction passed through unscented transform
    unscented_transform(urts, "obs")
                    
    log_likelihood_current_obs = loglikelihood(urts, z)
    push!(urts.ll, log_likelihood_current_obs)
        
    urts.SI = inv(urts.S)

    # compute cross variance of the state and the measurements
    Pxz = cross_variance(urts)
        
    #println("second print update", urts.zp, urts.S, Pxz, urts.SI)
        
    urts.K = Pxz * urts.SI        # Kalman gain
    urts.y = z .- urts.zp         # residual

    # update Gaussian state estimate (x, P)
    #println("critical point",urts.K , urts.y, urts.x)
    #urts.x = mod2pi.(urts.x + urts.K * urts.y)
    urts.x = urts.x + urts.K * urts.y
    urts.P = urts.P - urts.K *( urts.S * urts.K')
    #println("third print update",urts.K , urts.y, urts.x, urts.P)
        
    #println("x after update", urts.x)
end
 
 
function cross_variance(urts::URTS)
    Pxz = zeros(size(urts.sigmas_f,2), size(urts.sigmas_h,2))
    N = size(urts.sigmas_f,1)
    for i in 1:N
        if CIRCULAR
            dx =  angular_subtract(urts.sigmas_f[i,:], urts.x) 
        else
            dx = urts.sigmas_f[i,:] .- urts.x
        end
        dz = urts.sigmas_h[i,:] .- urts.zp
        Pxz += urts.Wc[i,:] .* (dx .* dz')
    end
    return Pxz
end
    
    
function urts_smoother(urts::URTS, x_inf, p_inf; l_arg_f = [])

    n, dim_x = size(x_inf)

    
    dts = [urts.dt for i in 1:n]
    Qs = [urts.Q for i in 1:n]

    # smoother gain
    Ks = zeros(n, dim_x, dim_x)

    num_sigmas = urts.num_sigmas

    xs, ps, Pp = copy(x_inf), copy(p_inf), copy(p_inf)
    sigmas_f = zeros(num_sigmas, dim_x)

    for k in n-1:-1:1
        # create sigma points from state estimate, pass through state func
        sigmas = sigma_points(urts.α, urts.β, urts.κ, xs[k,:], ps[k,:,:])
        for i in 1:num_sigmas
            urts.sigmas_f[i,:] = urts.fx(sigmas[i,:], dts[k], l_arg_f...)
            unscented_transform(urts, "smooth")
        end
        xb, Pb = urts.x, urts.P
                
        # compute cross variance
        Pxb = 0
        for i in 1:num_sigmas
            if CIRCULAR
                y = angular_subtract(urts.sigmas_f[i,:], xb)  #urts.sigmas_f[i,:] .- xb
                z = angular_subtract(sigmas[i,:], x_inf[k,:]) #sigmas[i,:] .- x_inf[k,:]
            else
                y = urts.sigmas_f[i,:] .- xb
                z = sigmas[i,:] .- x_inf[k,:]
            end
                                        
            Pxb = Pxb .+ urts.Wc[i,:] .* (z .* y')
        end

        # compute gain
        K = Pxb * inv(Pb)

        # update the smoothed estimates
        if CIRCULAR
            xs[k,:] .+=  K * angular_subtract(xs[k+1,:], xb) #  #
        else
             xs[k,:] .+= K * (xs[k+1,:] .- xb )
        end
        ps[k,:,:] .+=  K * (ps[k+1,:,:] .- Pb) * K'
        Ks[k,:,:] = K
        Pp[k,:,:] = Pb            
    end
     
    #for k in n-1:-1:1
    #    #prevent divergence of the covariance matrix
    #    for i in 1:size(ps[k,:,:],1)
    #        if ps[k,i,i]<0
    #            println("coucou", ps[k,:,:], ps[k+1,:,:])
    #            ps[k,:,:] = ps[k+1,:,:]
    #            break
    #        end
    #    end
    #end

    return (xs, ps, Ks, Pp)    
end
        

function joint_distribution(urts::URTS, Ps, x, P, K, Pp)
    
    a_jP_x =  Array{Float64}(undef, 0, 2, urts.dim_x) # dim t, dim joint_t, dim_x
    a_jP_p = Array{Float64}(undef, 0, 2*urts.dim_x, 2*urts.dim_x)
     

    for k in 1:size(x,1)-1

            
        P2 = Ps[k,:,:].-(K[k,:,:] * Pp[k,:,:]) * K[k,:,:]'


        a_jP_x = vcat(a_jP_x, reshape([x[k+1,:]'; x[k,:]'], (1,urts.dim_x, urts.dim_x))   )


        new_p = [ [ P[k+1,:,:] P[k+1,:,:]*K[k,:,:]' ] ;
                [ K[k,:,:] * P[k+1,:,:]  (K[k,:,:] * P[k+1,:,:]) * K[k,:,:]' .+ P2 ] ]
        new_p = reshape( new_p, (1,size(new_p)...))
        a_jP_p = vcat(a_jP_p, new_p)
    end
    return a_jP_x, a_jP_p
end
    
function loglikelihood(urts::URTS, z)
    #to be called at the end of update step
    return -0.5 * ( log(2*pi*norm(urts.S)) .+ (z-urts.zp)' * inv(urts.S) * (z-urts.zp) )
end
    
function total_ll(urts::URTS)
    return sum(urts.ll)
end
         
    
    
########################## OTHER USEFUL FUNCTIONS ##########################
#compute sigma_points
function sigma_points(α, β, κ, x, P)

    n = size(x,1)
    lambda_ = α^2 * (n + κ) - n
    #println("cholesky",(lambda_ + n).*P)
    P  = round.(P .* 10^12 ) / 10^12 
    
    U = cholesky((lambda_ + n).*P,  check = false).U 
    #println(P)
    #U = sqrt((lambda_ + n).*P)
    #println("U",U)
    sigmas = zeros(2*n+1, n)
    sigmas[1,:] = x#mod2pi.(x)#x
    
    for k in range(1,n)
        #if CIRCULAR
        #    sigmas[k+1,:]   = angular_subtract(x, .- U[k,:])
        #    sigmas[n+k+1,:] = angular_subtract(x,  U[k,:])
        #else
            sigmas[k+1,:]   = x .+ U[k,:]
            sigmas[n+k+1,:] = x .-  U[k,:]
        #end
        
    end
    return sigmas
end

    
function angular_subtract(x,y)
    x, y = real.(x), real.(y)
    z = (x .- y).%(2*π)
    for i in 1:size(z,1)
        if z[i] > π
            z[i] -= 2*π
        elseif z[i] < -π
            z[i] += 2*π
        end
    end
    return z
end
    
#comptue weights of the sigma points
function compute_weights(dim_x, α, β, κ)

    lambda_ = α^2 * (dim_x + κ) - dim_x

    c = .5 / (dim_x + lambda_)
    Wc = fill(c, 2*dim_x + 1)
    Wm = fill(c, 2*dim_x + 1)
    Wc[1] = lambda_ / (dim_x + lambda_) + (1 - α^2 + β)
    Wm[1] = lambda_ / (dim_x + lambda_)
    return (Wc, Wm)
end

#mean of a weighted angular variable #TO BE REDEFINED WHEN THE SYSTEM CHANGES
function sigma_angular_mean(sigmas, Wm)
    x = zeros(size(sigmas,2))
    sum_sin_1 = 0.
    sum_cos_1 = 0.
    sum_sin_2 = 0.
    sum_cos_2 = 0.
    for i in 1:size(sigmas,1)
        s = sigmas[i,:]
        sum_sin_1 += sin(s[1])*Wm[i]
        sum_cos_1 += cos(s[1])*Wm[i]
        sum_sin_2 += sin(s[2])*Wm[i]
        sum_cos_2 += cos(s[2])*Wm[i]
    end
    x[1] = mod2pi(atan(sum_sin_1, sum_cos_1))
    x[2] = mod2pi(atan(sum_sin_2, sum_cos_2))
    return x
end