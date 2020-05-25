#smooth a signal u with parameter a between 0 and 1
function smooth(u, a)
    y = zeros(size(u,1))
    y[1] = (1-a)*u[1]
    for k=2:length(u) 
        y[k] = a*y[k-1] + (1-a)*u[k]
    end
    return y
end

#return extrema of a signal s
function return_maxima_indexes(s)
    l_extrema = []
    l_der = s[2:end]-s[1:end-1]
    last_sign = l_der[1]>=0 ? true : false
    for (idx, x) in enumerate(l_der)
        sign = x>=0 ? true : false
        if sign==last_sign
            continue
        else
            last_sign = sign
            push!(l_extrema, idx)
        end
    end
    #remove artifacts by looking for impossible periods
    l_to_del = []
    l_T = l_extrema[2:end]-l_extrema[1:end-1]
    mean_T = mean(l_T)
    for (idx, T) in enumerate(l_T)
        if T<mean_T/3
            push!(l_to_del, l_extrema[idx])
            push!(l_to_del, l_extrema[idx+1])
        end
    end
    l_extrema = [x for x in l_extrema if !(x in l_to_del)]
    return l_extrema
end
