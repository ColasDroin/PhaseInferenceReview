#plot coupling function
function plot_F(F; title_fig = " ", cmap = "bwr", vmin = -0.3, vmax = 0.3, interpolation = "bessel", save = false, path = "")
    figure()
    imshow(permutedims(F, (2,1)), origin = "lower", interpolation = interpolation , cmap = cmap, vmin = vmin, vmax = vmax)
    colorbar()
    xlabel("θ")
    ylabel("ϕ")
    title(title_fig)
    if save
        if path==""
            savefig("Plots/Couplings/"*title_fig*".pdf")
        else
            savefig(path*title_fig*".pdf") 
        end
    end
    #show()
end


#plot simulated trace
function plot_simulated_trace(trace, tspan, path; split = false)
    if !split
        figure(figsize = (20,10))
        plot(tspan, trace[:,3], label = L"$y_{θ_t}$" , color = "C0")
        plot(tspan, trace[:,4], label = L"$y_{ϕ_t}$", color = "C1")
        plot(tspan, mod2pi.(trace[:,1])./(2*π), linestyle = "--", label = L"$θ_t$", color = "C0")
        plot(tspan, mod2pi.(trace[:,2])./(2*π), linestyle = "--", label = L"$ϕ_t$", color = "C1")
        xlabel("Time")
        legend()
        savefig(path*"Simulated_trace_example.pdf")
        show()
    else
        figure(figsize = (20,6))
        plot(tspan, trace[:,3], label = L"$y_{θ_t}$" , color = "C0")
        plot(tspan, mod2pi.(trace[:,1])./(2*π), linestyle = "--", label = L"$θ_t$", color = "C0")
        xlabel("Time")
        legend()
        savefig(path*"Simulated_trace_theta_example.pdf")
        show()
            
        figure(figsize = (20,6))
        plot(tspan, trace[:,4], label = L"$y_{ϕ_t}$", color = "C1")
        plot(tspan, mod2pi.(trace[:,2])./(2*π), linestyle = "--", label = L"$ϕ_t$", color = "C1")
        xlabel("Time")
        legend()
        savefig(path*"Simulated_trace_phi_example.pdf")
        show()
                 
    end
end



#plot fitted trace
function plot_fitted_trace(trace, x, p, Theta, Phi)
    tspan = 1:size(x,1)   
    figure(figsize=(20,10))
    plot(tspan, trace[:,3], ".",label = "Data theta", color = "C0", alpha = 1)
    plot(tspan, mod2pi.(trace[:,1])/(2*π), "--",label = "Phase theta",  color = "C0", alpha = 0.5)
    plot(tspan,  ω_fast(Theta, x[:,1]), label = "Fit theta", color = "C1", alpha = 1)
    plot(tspan,mod2pi.(x[:,1])/(2*π), "--",label = "Inferred theta",  color = "C1", alpha = 0.5)
    legend()
    xlabel("Time")
    ylabel("Signal")
    #savefig("Figures/signal_theta.pdf")
    show()

    figure(figsize=(20,10))
    plot(tspan, trace[:,4], ".",label = "Data phi", color = "C0", alpha = 1)
    plot(tspan, mod2pi.(trace[:,2])/(2*π), "--",label = "Phase phi",  color = "C0", alpha = 0.5)
    plot(tspan, ω_fast(Phi, x[:,2]), label = "Fit phi", color = "C1", alpha = 1)
    plot(tspan,mod2pi.(x[:,2])/(2*π), "--",label = "Inferred phi",  color = "C1", alpha = 0.5)
    legend()
    xlabel("time")
    ylabel("Signal")
    #savefig("Figures/signal_phi.pdf")
    show()
end

#plot waveform/coupling of the current variable
function plot_all_variable(ω_θ, ω_ϕ, F_θ, F_ϕ, Phi, Theta, domain_phase, tag, DEBUG)

end
        