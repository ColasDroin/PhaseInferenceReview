using TensorOperations
function compute_coupling_discretization(l_x, l_p, l_jP_x, l_jP_P, resolution, lambd, theta_f = Theta.f, phi_f = Phi.f, dt = dt )
    
    #println(typeof(l_jP_x))
    dim_x = size(l_x[1][1,:],1)
    
    F_theta_bad = zeros(Float64, resolution, resolution)
    F_phi_bad =   zeros(Float64, resolution, resolution)
    F_count_bad = zeros(Float64, resolution, resolution)
    
    F_theta = zeros(Float64, resolution, resolution)
    F_phi =   zeros(Float64, resolution, resolution)

    #create 2d domain with every couple of phase
    domain = range(0, stop=2*π, length=resolution+1)[1:end-1]
    domain_large = range(-2*π, stop=4*π, length=3*resolution+1)[1:end-1]
    domain_2d_large =  zeros(Float64, 3*resolution, 3*resolution, dim_x)
    for (idx_theta, theta) in enumerate(domain_large)
        for (idx_phi, phi) in enumerate(domain_large)
        domain_2d_large[idx_theta,idx_phi,:] = [theta, phi]
        end
    end

    
    
    flat_domain_2d_large = reshape(domain_2d_large, size(domain_2d_large,1)*size(domain_2d_large,2),dim_x )
    
    #transition matrices
    Mat_tr_theta = [  mod2pi(theta+theta_f*dt) for theta in domain, phi in domain ]
    Mat_tr_phi   = [  mod2pi(phi+phi_f*dt) for theta in domain, phi in domain ]
        
    #create empty array for estimating angles
    Mat_theta_end = zeros(ComplexF64, resolution,resolution)
    Mat_phi_end = zeros(ComplexF64, resolution,resolution)
    
    Mat_theta_end_real = zeros(Float64, resolution,resolution)
    Mat_phi_end_real = zeros(Float64, resolution,resolution)
    
    Mat_norm = zeros(Float64, resolution,resolution)

    Mat_theta_end_large = zeros(ComplexF64, 3*resolution,3*resolution)
    Mat_phi_end_large =   zeros(ComplexF64, 3*resolution,3*resolution)
    Mat_norm_large = zeros(Float64, 3*resolution,3*resolution)
    Mat_cond_mu_large = zeros(Float64, 3*resolution,3*resolution) #TO DELETE ?

    
    for (x, p, jP_x, jP_P) in zip(l_x, l_p, l_jP_x, l_jP_P)
        for t in 1:size(x,1)-1
                    
            
            #compute the conditional normal distribution of theta_tdt and phi_tdt given theta_t and phi_t            
            A = jP_P[t,1:2,3:4] * inv(jP_P[t,3:4,3:4])
            B = domain_2d_large[:,:,:] .- reshape(mod2pi.(jP_x[t,2,:]),1,1,size(jP_x[t,2,:],1))
            
            @tensor begin
            C[1,2,3] := A[3,4]*B[1,2,4]
            end
            
            
            Mat_cond_mu_large = reshape(mod2pi.(jP_x[t,1,:]),1,1, size(mod2pi.(jP_x[t,1,:]),1),1) .+ C
            
            Mat_cond_sigma = jP_P[t,1:2,1:2]- (jP_P[t,1:2,3:4] * inv(jP_P[t,3:4,3:4])) * jP_P[t,3:4,1:2]

            #compute probability of being at a given state using wrapped normal law
            
            #ugly rouding but can't do better with Julia 1.0
            #println(p[t,:,:])
            flat_p_x_large = pdf( MvNormal( mod2pi.(x[t,:]),  round.(p[t,:,:] .* 10^8 ) / 10^8 ) , flat_domain_2d_large' )  
            Mat_p_x_large = reshape(flat_p_x_large, size(domain_2d_large,1),size(domain_2d_large,2) ) 
            Mat_p_x_large = Mat_p_x_large ./ sum(Mat_p_x_large)

            Mat_theta_end_large .+= exp.(1im .* Mat_cond_mu_large[:,:,1] .- Mat_cond_sigma[1,1] ./ 2) .* Mat_p_x_large
            Mat_phi_end_large .+= exp.(1im .* Mat_cond_mu_large[:,:,2] .- Mat_cond_sigma[2,2] ./ 2) .* Mat_p_x_large
            Mat_norm_large .+= Mat_p_x_large
        end
        
        #println(Mat_theta_end_large[25,25])
        #println(Mat_norm_large[25,25])
        
        #println(x)
        #println(size(x))
        v =   angle.(exp.(1im.*(x[2:end,:] .- x[1:end-1,:])))   ./ dt
        for (ph1, ph2,v1,v2) in zip(x[1:size(v,1),1],x[1:size(v,1),2],v[:,1],v[:,2])
            ph1_idx::Int64 = round( mod2pi.(ph1)/(2*π)*resolution)%(resolution)
            ph2_idx::Int64 = round( mod2pi.(ph2)/(2*π)*resolution)%(resolution)

            F_theta_bad[ph1_idx+1, ph2_idx+1] += v1
            F_phi_bad[ph1_idx+1, ph2_idx+1] += v2 
            F_count_bad[ph1_idx+1, ph2_idx+1] += 1
        end
        
    end

    #sum wrapped distribution
    for idx_periodicity_theta in 0:2
        for idx_periodicity_phi in 0:2
            Mat_theta_end .+= Mat_theta_end_large[resolution*idx_periodicity_theta+1:resolution*(idx_periodicity_theta+1),resolution*idx_periodicity_phi+1:resolution*(idx_periodicity_phi+1)]
            Mat_phi_end .+= Mat_phi_end_large[resolution*idx_periodicity_theta+1:resolution*(idx_periodicity_theta+1),resolution*idx_periodicity_phi+1:resolution*(idx_periodicity_phi+1)]
            Mat_norm .+= Mat_norm_large[resolution*idx_periodicity_theta+1:resolution*(idx_periodicity_theta+1),resolution*idx_periodicity_phi+1:resolution*(idx_periodicity_phi+1)]
        end
    end
    
    
    #println(Mat_theta_end[1,:])
    #compute coupling
    Mat_theta_end_real = mod2pi.(angle.(Mat_theta_end ./ Mat_norm))
    Mat_phi_end_real = mod2pi.(angle.(Mat_phi_end ./ Mat_norm))
    #println(Mat_theta_end)

    #diff angle
    Mat_diff_theta = Mat_theta_end_real .- Mat_tr_theta
    Mat_diff_phi = Mat_phi_end_real .- Mat_tr_phi

    #correct for wrong values
    for idx_theta in 1:resolution
        for idx_phi in 1:resolution
            if Mat_diff_theta[idx_theta, idx_phi] < -π
                Mat_diff_theta[idx_theta, idx_phi] += 2*π
            elseif Mat_diff_theta[idx_theta, idx_phi] > π
                Mat_diff_theta[idx_theta, idx_phi] -= 2*π
            end

            if Mat_diff_phi[idx_theta, idx_phi] < -π
                Mat_diff_phi[idx_theta, idx_phi] += 2*π
            elseif Mat_diff_phi[idx_theta, idx_phi]>π
                Mat_diff_phi[idx_theta, idx_phi] -= 2*π
            end
        end
    end
    
    #compute final coupling

    F_theta = Mat_diff_theta ./ dt
    F_phi = Mat_diff_phi ./ dt
    
    #regularize
    F_theta = F_theta .* Mat_norm .* dt
    F_phi = F_phi .* Mat_norm .* dt
    
    F_theta = F_theta ./ (Mat_norm .* dt .+ lambd .+ 10^-15)
    F_phi = F_phi ./ (Mat_norm .* dt .+ lambd .+ 10^-15)
 
    
    F_theta_bad = F_theta_bad ./ F_count_bad
    F_phi_bad = F_phi_bad ./ F_count_bad
    F_theta_bad = F_theta_bad .- Theta.f
    F_phi_bad = F_phi_bad .- Phi.f
    

    #get fourier transform
    FT_theta = fft(F_theta) 
    FT_phi = fft(F_phi)
    
    return F_theta, F_phi, F_theta_bad, F_phi_bad, Mat_norm, FT_theta, FT_phi
    
end
        
        
function f_hat(x, n_harm)
    """
    vect_exp = (1/n_harm^2).*[ exp(1im * ( k*x[1] + l*x[2] )) for k in 0:n_harm-1 for l in 0:n_harm-1 ] 
    f_hat = vcat( [ 1, x... ], vect_exp )
    """
    
    """
    vect_cos = (1/n_harm^2).*[ cos( k*x[1] + l*x[2] ) for k in 0:n_harm-1 for l in 0:n_harm-1 ]
    vect_sin =  (1/n_harm^2).*[ -sin( k*x[1] + l*x[2] ) for k in 0:n_harm-1 for l in 0:n_harm-1 ]
    f_hat = vcat( [ 1, x... ], vect_cos, vect_sin )
    """
    
    """
    vect_cos = (1/n_harm^2).*[ cos( k*x[1] + l*x[2] ) for k in 0:n_harm-1 for l in 0:n_harm-1 ]
    vect_sin =  (1/n_harm^2).*[ -sin( k*x[1] + l*x[2] ) for k in 0:n_harm-1 for l in 0:n_harm-1 ]
    f_hat = vcat( [ 1, x... ], vect_cos )
    """
    
    f_hat = [ 1, x... ]
    return f_hat
end



function compute_coupling_fourier_in_loop(α, β, κ, xt, pt, jP_xt, jP_Pt, Wm, jWm, dim_x, n_harm, sigmas_f_hat, sigmas_f_hat_c)
    #compute PHI (only until T-1, and starting from k=1 instead of k=0)

    #sigmas_phi = sigma_points(α, β, κ, xt, pt)


    #for i in 1:size(sigmas_phi,1)
    #    sigmas_f_hat[i,:] = f_hat(sigmas_phi[i,:], n_harm)
    #end
    #PHI_k = sigmas_f_hat' * (Wc .* sigmas_f_hat)
    
    
    ###correct phase
    mod_jP_xt = mod2pi.(jP_xt)
    for i in 1:2
        if abs(mod_jP_xt[1,i]-mod_jP_xt[2,i])>π
            mod_jP_xt[1,i] =  mod_jP_xt[2,i] +  jP_xt[1,i]-jP_xt[2,i]
        end
    end
    
    #println(mod_jP_xt)
    
    #compute C (starting from k=1 instead of k=0)
    #sigmas_c = sigma_points(α, β, κ, hcat(jP_xt...)', jP_Pt)
    sigmas_c = sigma_points(α, β, κ, hcat(mod_jP_xt...)', jP_Pt)
    
    
    
    for i in 1:size(sigmas_c,1)
        sigmas_f_hat_c[i,:] = f_hat(sigmas_c[i,3:4], n_harm)
    end
    
    
    #println(sigmas_f_hat_c)
    #println(jWm)
    
    PHI_k = sigmas_f_hat_c' * (jWm .* sigmas_f_hat_c)
    
    #println(PHI_k)
    
    #C_k = zeros(ComplexF64, dim_x, size(sigmas_f_hat_c,2))
    C_k = zeros(Float64, dim_x, size(sigmas_f_hat_c,2))
    N = size(sigmas_f_hat_c,1)
    for i in 1:N
        dx = sigmas_c[i,1:2]
        fx = sigmas_f_hat_c[i,:]
        C_k += jWm[i] .* (dx .* fx')
    end
    
    return PHI_k, C_k
end


function compute_coupling_fourier_in_loop_discrete(xt, pt, jP_xt, jP_Pt, dim_x, n_harm, flat_domain_2d_large, flat_domain_4d_large, flat_f_hat_prod, flat_f_hat_x_prod )
    println("start")
    ###try to compute the expectation using the old way
    flat_Mat_P = pdf( MvNormal( mod2pi.(xt),  round.(pt .* 10^4 ) / 10^4 ) , flat_domain_2d_large' )
    flat_Mat_P_double = pdf( MvNormal( mod2pi.( reshape(jP_xt',dim_x*2) ),  round.(jP_Pt .* 10^4 ) / 10^4 ) , flat_domain_4d_large' )
    Mat_P_double = reshape(flat_Mat_P_double, size(flat_domain_2d_large,1), size(flat_domain_2d_large,1))
    println("middle")
    @tensor begin
    bi_flat_Mat_P[a,b] := flat_Mat_P[a]*flat_Mat_P[b]
    PHI_k[b,d] := flat_f_hat_prod[a,b,c,d]*bi_flat_Mat_P[a,c]
    C_k[d,b] := flat_f_hat_x_prod[a,b,c,d]*Mat_P_double[c,a]
    end
    println("end")
    return PHI_k, C_k
end




function compute_coupling_fourier(FT_θ, FT_ϕ, n_harm, l_x, l_p, l_jP_x, l_jP_P, α, β, κ, lambd, theta_f = Theta.f, phi_f = Phi.f, dt = dt )
    #println("OK")
    ### TEST BEFORE
    
    #A = [ hcat([theta_f*dt 1 0], reshape(FT_θ', size(FT_θ,1)* size(FT_θ,2) )' .* dt) ; 
    #      hcat([phi_f*dt 0 1], reshape(FT_ϕ', size(FT_ϕ,1)* size(FT_ϕ,2) )' .* dt)  ]
    
    #A = [ hcat([theta_f*dt 1 0], real.(reshape(FT_θ', size(FT_θ,1)* size(FT_θ,2) ))' .* dt, imag.(reshape(FT_θ', size(FT_θ,1)* size(FT_θ,2) ))' .* dt) ; 
    #      hcat([phi_f*dt 0 1], real.(reshape(FT_ϕ', size(FT_ϕ,1)* size(FT_ϕ,2) ))' .* dt, imag.(reshape(FT_ϕ', size(FT_ϕ,1)* size(FT_ϕ,2) ))' .* dt)  ]
    
    #A = [ hcat([theta_f*dt 1 0], real.(reshape(FT_θ', size(FT_θ,1)* size(FT_θ,2) ))' .* dt) ; 
    #      hcat([phi_f*dt 0 1], real.(reshape(FT_ϕ', size(FT_ϕ,1)* size(FT_ϕ,2) ))' .* dt)  ]
    
    A = [ [theta_f*dt 1 0] ; [phi_f*dt 0 1]  ]
    
    
    
    t = [2, 1]
    t_dt = A*f_hat(t, n_harm)
    println("BEFORE ", mod2pi.(real.(t_dt)))
    
    
    dim_x = size(l_x[1][1,:],1)
    dim_f_hat = 1 + dim_x #+ n_harm * n_harm #*2 #remove *2 if work with complex
    
    #PHI = zeros(ComplexF64, dim_f_hat, dim_f_hat ) 
    #C =  zeros(ComplexF64, dim_x, dim_f_hat )
    PHI = zeros(Float64, dim_f_hat, dim_f_hat )
    C =  zeros(Float64, dim_x, dim_f_hat )
    
    num_sigmas = 2*dim_x + 1
    num_sigmas_c = 4*dim_x + 1
    
    #sigmas_f_hat = zeros(ComplexF64, num_sigmas, dim_f_hat)
    #sigmas_f_hat_c = zeros(ComplexF64, num_sigmas_c, dim_f_hat)
    sigmas_f_hat = zeros(Float64, num_sigmas, dim_f_hat)
    sigmas_f_hat_c = zeros(Float64, num_sigmas_c, dim_f_hat)
                    
    #compute sigma weights
    (Wc, Wm) = compute_weights(dim_x, α, β, κ)
    
    #compute sigma weights
    (jWc, jWm) = compute_weights(dim_x*2, α, β, κ)
    
    
    T = 0
    
    """
    ### OLD WAY
    resolution = 20
    domain_large = range(-2*π, stop=4*π, length=3*resolution+1)[1:end-1]
    domain_2d_large =  zeros(Float64, 3*resolution, 3*resolution, dim_x)
    domain_4d_large = zeros(Float64, 3*resolution, 3*resolution, 3*resolution, 3*resolution, dim_x*2)
    f_hat_large = zeros(size(domain_2d_large,1), size(domain_2d_large,2), dim_f_hat)
    for (idx_theta, theta) in enumerate(domain_large)
        for (idx_phi, phi) in enumerate(domain_large)
            domain_2d_large[idx_theta,idx_phi,:] = [theta, phi]
            f_hat_large[idx_theta,idx_phi,:] = f_hat( [theta, phi], n_harm)
            for (idx_theta2, theta2) in enumerate(domain_large)
                for (idx_phi2, phi2) in enumerate(domain_large)
                    domain_4d_large[idx_theta,idx_phi,idx_theta2,idx_phi2,:] = [theta, phi, theta2, phi2]
                end
            end
        end
    end
    
    flat_domain_2d_large = reshape(domain_2d_large, size(domain_2d_large,1)*size(domain_2d_large,2),dim_x )
    flat_domain_4d_large = reshape(domain_4d_large, size(domain_2d_large,1)^2*size(domain_2d_large,2)^2,dim_x*2 )
    flat_f_hat_large = reshape(f_hat_large, size(domain_2d_large,1)*size(domain_2d_large,2),dim_f_hat )
    @tensor begin
    flat_f_hat_prod[1,2,3,4] := flat_f_hat_large[1,2]*flat_f_hat_large[3,4]
    flat_f_hat_x_prod[1,2,3,4] :=  flat_f_hat_large[1,2]*flat_domain_2d_large[3,4]
    end
    """
    

    
    for (x, p, jP_x, jP_P) in zip(l_x, l_p, l_jP_x, l_jP_P)
        for k in 1:size(x,1)-1            
            PHI_k, C_k = compute_coupling_fourier_in_loop(α, β, κ, x[k,:], p[k,:,:], jP_x[k,:,:], jP_P[k,:,:], Wm, jWm, dim_x, n_harm,sigmas_f_hat, sigmas_f_hat_c)
            #PHI_k, C_k = compute_coupling_fourier_in_loop_discrete( x[k,:], p[k,:,:], jP_x[k,:,:], jP_P[k,:,:], dim_x, n_harm, flat_domain_2d_large, flat_domain_4d_large,flat_f_hat_prod, flat_f_hat_x_prod)
            
            PHI = PHI .+ PHI_k
            C = C .+ C_k
            T=+1
        end
    end
    PHI = PHI ./ T
    C = C ./ T

    println("PHI", PHI)
    println("C", C)
    #println(PHI)
    #println(C)
    A = C * inv(PHI)
    #println(inv(PHI))
    
    #FT_theta = reshape(A[1, 1 + dim_x + 1 : end ], n_harm, n_harm)
    #FT_phi = reshape(A[2, 1 + dim_x + 1 : end ], n_harm, n_harm)
    FT_theta = 0#reshape(A[1, 1 + dim_x + 1 : end ], n_harm, n_harm)
    FT_phi = 0 #reshape(A[2, 1 + dim_x + 1 : end ], n_harm, n_harm)
    
    ### TEST
    println(A)
    x = [2., 1.]
    x_dt = A*f_hat(x, n_harm)
    #println(x_dt)
    println("AFTER", mod2pi.(x_dt))
    
    return FT_theta, FT_phi
end