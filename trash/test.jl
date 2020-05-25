mutable struct URTS
    x::Vector{Float64} #state estimate vector
    P::Matrix{Float64} #covariance estimate matrix
    R::Matrix{Float64} #measurement noise matrix
    Q::Matrix{Float64} #process noise matrix
    K::Vector{Float64} #Kalman gain
    y::Vector{Float64} #innovation residual
    z::Matrix{Float64} #measurment
    S::Matrix{Float64} # system uncertainty
    SI::Matrix{Float64} # inverse system uncertainty

    _dim_x::Int32
    _dim_z::Int32
    points #TO BE DEFINED
    _num_sigmas #TO BE DEFINED
    dt::Float64
    fx
    hx

    # weights for the means and covariances.
    Wm #TO DEFINE
    Wc #TO DEFINE

    sigmas_f::Matrix{Float64}
    sigmas_h::Matrix{Float64}

end

"""
URTS constructor
"""
function URTS(dim_x, dim_z, dt, hx, fx, points):
    x = zeros(dim_x)
    P = eye(dim_x)
    Q = eye(dim_x)
    R = eye(dim_z)
    _dim_x = dim_x
    _dim_z = dim_z
    points_fn = points
    _num_sigmas = points.num_sigmas()

    # weights for the means and covariances.
    Wm, Wc = points.Wm, points.Wc

    residual_x = subtract
    residual_z = subtract

    sigmas_f = zeros((_num_sigmas, _dim_x))
    sigmas_h = zeros((_num_sigmas, _dim_z))

    K = zeros((dim_x, dim_z))    # Kalman gain
    y = zeros((dim_z))           # residual
    z = array([[None]*dim_z]).T  # measurement
    S = zeros((dim_z, dim_z))    # system uncertainty
    SI = zeros((dim_z, dim_z))   # inverse system uncertainty

    return URTS(x, P, R, Q, K, y, z, S, SI, _dim_x, _dim_z, points, _num_sigmas, dt, fx, hx, Wm, Wc, sigmas_f, sigmas_h)
