
"""
This module contains code for optimal networks
"""
module NetworkOptimization

include("OscillatorNetworks.jl")

using LightGraphs, DifferentialEquations
using Plots
using NLsolve
using Optim
using ReverseDiff
using SpecialFunctions
using .OscillatorNetworks

""" Return the total variance for white noise inputs
for corrected susceptances b → b cos(δ)
"""
function total_variance_white_noise(topol::OscillatorNetworks.NetworkTopology, R::Matrix, b::Array)
    L = topol.E*spdiagm(b)*topol.E'

    X = (L .+ 1.0/size(L)[1]) \ R
    return trace(X)/(2topol.γ)
end

""" Return the mean power flows along the edges given the
correlation matrix R.

This corresponds to the optimization RHS for white noise
close to synchrony, i.e., with Eᵀ δ_ss ≈ 0.
"""
function white_noise_synchronous(topol::OscillatorNetworks.NetworkTopology, b::Array, α::Float64, R::Matrix)
#     cosines = cos.(topol.E'*OscillatorNetworks.ss_power_angles(b, topol))
#     println(cosines)
    L = topol.E*spdiagm(b)*topol.E'

    ELinv = (L .+ 1.0) \ topol.E

    return sum(ELinv.*(R*ELinv), 1)[1,:]
    # return diag(ELinv'*R*ELinv)
end

struct OrnsteinUhlenbeckCorrelations
    R::Matrix    # spatial correlation matrix
    κ::Float64   # inverse correlation timescale
end

""" Total variance of the fluctuations under Ornstein-Uhlenbeck noise
"""
function total_variance_ornstein_uhlenbeck(topol::OscillatorNetworks.NetworkTopology, C::OrnsteinUhlenbeckCorrelations, b::Array)
    L = topol.E*spdiagm(b)*topol.E'
    n = size(L)[1]

    Laug = L + C.κ*(C.κ + topol.γ)*speye(topol.verts)

    X = Laug \ ((L .+ 1.0/n) \ C.R)
    return trace(X)*(C.κ + topol.γ)/topol.γ
end


""" Return the partial derivatives of the fluctuation average by the susceptances
given Ornstein-Uhlenbeck noise in time and close to synchrony,
i.e., Eᵀ δ_ss ≈ 0.
"""
function ornstein_uhlenbeck_synchronous(topol::OscillatorNetworks.NetworkTopology, b::Array, α::Float64, C::OrnsteinUhlenbeckCorrelations)
    L = topol.E*spdiagm(b)*topol.E'
    n = size(L)[1]

    LdagE = (L .+ 1.0/n) \ topol.E # the additional term 1 1^T E = 0

    L_aug = C.κ*(C.κ + topol.γ)*speye(topol.verts) + L

    L_fact = cholfact(L_aug)
    LaugR = L_fact \ C.R
    LaugE = L_fact \ topol.E

    # pd = LdagE'*LaugR*LdagE + LaugE'*LaugR'*LdagE
    # return -diag(pd)

    # efficient
    ppd = -sum(LdagE.*(LaugR*LdagE) .+ LaugE.*(LaugR'*LdagE), 1)[1,:]
    return ppd
end

""" Run fixed point optimization on the given
network topology with given parameters.
the function rhs_func has the signature

    rhs_func(topol, b, α, f_args...)

The network optimization happens close to synchrony,
i.e., all the Eᵀ δ_ss ≈ 0.
"""
function optimize_synchronous_network(topol::OscillatorNetworks.NetworkTopology,
    rhs_func, f_args,
    α::Float64, b₀::Array, b_total::Float64;
    tolerance::Float64=1e-6, max_iter::Int64=10000,
    intermediate_diffs::Bool=false, intermediate_steps::Int64=25)
    # run fixed point optimization
    b = b₀
    b *= (b_total/sum(b.^α))^(1.0/α)

    change = 1.0
    rhs = similar(b)
    for i=1:max_iter
        rhs = rhs_func(topol, b, α, f_args...)

        b_new = abs.((b.^2).*rhs).^(1.0/(1.0 + α))
        b_new *= (b_total/sum(b_new.^α))^(1.0/α)

        change = norm(b - b_new)
        b = b_new

        if intermediate_diffs && ((i % intermediate_steps) == 0)
            println("i: $(i)\t\t|b_new - b_old| = $(change)")
        end
        if change < tolerance
            break
        end
    end

    return b, change
end

""" Return the mean power flows along the edges given the
correlation matrix R.
"""
function white_noise_general(topol::OscillatorNetworks.NetworkTopology, b::Array, α, δ₀, R)
    # steady state angles and Laplacian
    δ_cur, res = OscillatorNetworks.ss_power_angles(b, topol; iterations=2, δ₀=δ₀,
        print_stats=false, return_res=true)
    δ_e = topol.E'*δ_cur

    cosines = cos.(δ_e)
    sins = sin.(δ_e)

    L = topol.E*spdiagm(b.*cosines)*topol.E'

    # S matrix
    # ELinv = topol.E'*pinv(L) # Below: more efficient version exploiting nullspaces
    ELinv = (L .+ 1.0) \ topol.E
    ELinv = ELinv'
    S = ELinv*topol.E

    Δε² = sum((ELinv*R).*ELinv, 2)[:,1]
    # Δε² = diag(ELinv*R*ELinv')

    # println(norm(Δε² - diag(ELinv*R*ELinv')))

    correction = sins.*(S*(b.*sins.*Δε²))

    rhs = cosines.*Δε² + correction

    return rhs, cosines, δ_cur, res
end

""" Run fixed point optimization on the given
network topology with given parameters.
the function rhs_func has the signature

    rhs, cosines, δ_new, res = rhs_func(topol, b, α, δ_cur, f_args...),

where cosines is the vector of cos(Eᵀ δ_new)

The network optimization is completely general,
whether we are close to synchrony or not.
"""
function optimize_general_network(topol::OscillatorNetworks.NetworkTopology,
    rhs_func, f_args,
    α::Float64, b₀::Array, b_total::Float64;
    tolerance::Float64=1e-6, max_iter::Int64=10000,
    intermediate_diffs::Bool=false, intermediate_steps::Int64=10)

    # run fixed point optimization
    b = b₀
    b *= (b_total/sum(b.^α))^(1.0/α)

    cosines = nothing
    δ_cur = OscillatorNetworks.ss_power_angles(b, topol; iterations=2)
    rhs = nothing

    change = 1.0
    change_δ = 1.0
    res = nothing
    for i=1:max_iter
        rhs, cosines, δ_new, res = rhs_func(topol, b, α, δ_cur, f_args...)

        # if length(rhs[rhs .< 0]) > 0
        #     println(rhs[rhs .< 0])
        # end

        # set negative entries to zero to skip over partially/unconverged steps
        rhs_proj = copy(rhs)
        rhs_proj[rhs_proj .< 0.0] = 0.0

        b_new = ((b.^2).*rhs_proj).^(1.0/(1.0 + α))
        b_new *= (b_total/sum(b_new.^α))^(1.0/α)

        change = norm(b - b_new)
        change_δ = norm(rem2pi.(δ_cur - δ_new, RoundNearest))
        b = b_new
        δ_cur = δ_new

        if intermediate_diffs && ((i % intermediate_steps) == 0)
            println("i: $(i)\t\t|b_cur - b_new|: $(change)\t||δ_cur - δ_new||: $(change_δ)")
        end

        if change < tolerance
            # print("$(change_δ)")
            break
        end
    end

    return b, δ_cur, change, change_δ, res
end

end
