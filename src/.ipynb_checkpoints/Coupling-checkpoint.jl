#build coupling from Gaussian
function build_F_from_2D_gaussian(l_v_mu, l_mat_sigma, l_amp, v_domain)
    flat_v_res = zeros( size(v_domain,1)*size(v_domain,1))
    mat_domain =  [ [theta, phi] for theta in v_domain, phi in v_domain ] 
    flat_mat_domain = reshape(mat_domain, size(mat_domain,1)*size(mat_domain,2) )
    flat_mat_domain = hcat(flat_mat_domain...) #flatten over last dimension
    for (v_mu, mat_sigma, amp) in zip(l_v_mu, l_mat_sigma, l_amp)
        flat_v_res += amp * pdf( MvNormal(  v_mu, mat_sigma), flat_mat_domain)
        flat_v_res += amp * pdf( MvNormal( [v_mu[1]+2*π, v_mu[2]], mat_sigma), flat_mat_domain)
        flat_v_res += amp * pdf( MvNormal( [v_mu[1], v_mu[2]+2*π], mat_sigma), flat_mat_domain)
        flat_v_res += amp * pdf( MvNormal( [v_mu[1]+2*π, v_mu[2]+2*π], mat_sigma), flat_mat_domain)
    end
    return reshape(flat_v_res, (size(mat_domain,1),size(mat_domain,2)) )
end
