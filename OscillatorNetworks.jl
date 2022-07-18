"""
This module contains code for oscillator networks
"""

module OscillatorNetworks

using LightGraphs, DifferentialEquations
using Plots
using NLsolve
using Optim
using ReverseDiff
using SpecialFunctions
using Statistics
using LinearAlgebra

struct NetworkTopology
    verts::Int64
    edges::Int64
    points::Array
    g::Graph
    E::Matrix
    γ::Float64
    P_ss::Vector
end

""" Return the number of loops in b and the maximal number of loops
given the topology topol.
"""
function loops(topol::NetworkTopology, b; thr=1e-6)
    edges = sum(b .> thr)

    loops = topol.edges - topol.verts + 1
    return edges - topol.verts + 1, loops
end

function construct_topology(points, γ)
    g, weights = euclidean_graph(Matrix(points'), cutoff=1.01/(sqrt(size(points)[1]) - 1))
    E = incidence_matrix(g, oriented=true)

    # random steady state input with zero mean and scaled by number of oscillators
    P_ss = randn(nv(g))
    P_ss .-= mean(P_ss)

    # scale by number of vertices
    P_ss ./= nv(g)

    return NetworkTopology(nv(g), ne(g), points, g, E, γ, P_ss)
end

function draw_network!(p, topol::NetworkTopology, b; color=colorant"black")
    points = topol.points

    for (e, width) = zip(edges(topol.g), b)
        x = points[e.src,:]
        y = points[e.dst,:]

        p = plot!(p, [x[1], y[1]], [x[2], y[2]], w=3width, c=color)
    end

    # plot output node
#     r = plot!(p, [points[1,1]], [points[1,2]], seriestype=:scatter, c=colorant"red", markersize=10)
    return p
end

function triang_grid_points(N::Integer)
    points = []

    b = sqrt(3.0)/2.0
    for i=0:N-1
        for j=0:N-1
            push!(points, [float(i + 0.5*mod(j, 2)), j*b])
        end
    end

    # turn Array of Arrays into a matrix
    points = vcat(points'...)

    return points/(N-1)
end

""" Return a triangular grid NetworkTopology with NxN nodes
"""
function triang_grid_network(N::Integer, γ::Float64)
    points = triang_grid_points(N)

    # construct network properties
    topol = construct_topology(points, γ)
    return topol
end

""" Return an array of points on an NxN square grid
in the unit square.
"""
function square_grid_points(N::Integer)
    points = []
    for i=0:N-1
        for j=0:N-1
            push!(points, float([i, j]))
        end
    end

    # turn Array of Arrays into a matrix
    points = vcat(points'...)

    return points/(N-1)
end

""" Return a square grid NetworkTopology with NxN nodes
"""
function square_grid_network(N::Integer, γ::Float64)
    points = square_grid_points(N)

    # construct network properties
    topol = construct_topology(points, γ)
    return topol
end

function ss_load_flow(delta_ss, b, topol)
    # load flow equations
    B = spdiagm(b)
    return -(topol.E*B*sin.(topol.E[2:end,:]'*delta_ss))[2:end] + topol.P_ss[2:end]
end

function ss_load_flow_jac(delta_ss, b, topol::NetworkTopology)
    # jacobian of the power flow equations wrt delta
            B_tilde = spdiagm(b.*cos.(topol.E[2:end,:]'*delta_ss))
    return -(topol.E*B_tilde*topol.E')[2:end,2:end]
end

function ss_power_angles(b::Array, topol::NetworkTopology; print_stats=false, iterations=100,
    δ₀=nothing, print_failure_msg=false, return_res=false)
    # calculate and return the steady state power angles
    # by nonlinear root finding
    # shift the angle at index 1 to zero

    if δ₀ == nothing
       δ₀ = zeros(topol.verts - 1)
    end

    # we shift to make root finding easier.
    if length(δ₀) == topol.verts
        δ₀ -= δ₀[1]
        δ₀ = δ₀[2:end]
    end

    res = nlsolve(x -> ss_load_flow(x, b, topol), x -> ss_load_flow_jac(x, b, topol),
        δ₀; inplace=false, iterations=iterations)

    if print_stats
        println(res)
    end

    if !converged(res) && print_failure_msg
        println("power angles did not converge!")
#         return zeros(verts)
    end

    delta_ss = zeros(topol.verts)
    delta_ss[2:end] = res.zero

    if return_res
        return rem2pi.(delta_ss, RoundNearest), res
    else
        # result is in [-π,π]
        return rem2pi.(delta_ss, RoundNearest)
    end
end

""" Return a random correlation matrix which is positive semidefinite
and whose row and column sums vanish.
Normalize such that the eigenvalues are between 0 and 1
"""
function random_correlation_matrix(topol::NetworkTopology; project=true)
    return random_correlation_matrix(topol.verts; project=project)
end

""" Return a random correlation matrix which is positive semidefinite
and whose row and column sums vanish.
Normalize such that the eigenvalues are between 0 and 1
"""
function random_correlation_matrix(verts::Integer; project=true)
    R = randn(verts, verts)

    # symmetric pos def
    R = R*R'

    # project onto subspace ⟂ 1
    if project
        P = I - ones(size(R))/size(R)[1]
        R = P*R*P
    end

    return R/tr(R)
end

""" Return a correlation matrix for maximally uncorrelated inputs.
This means projecting the unit matrix onto the subspace ⟂ 1⃗ and
normalizing the trace to 1 if desired.
"""
function uncorrelated_correlation_matrix(topol::NetworkTopology; project=true)
    return uncorrelated_correlation_matrix(topol.verts; project=project)
end

""" Return a correlation matrix for maximally uncorrelated inputs.
This means projecting the unit matrix onto the subspace ⟂ 1⃗ and
normalizing the trace to 1 if desired.
"""
function uncorrelated_correlation_matrix(verts::Integer; project=true)
    R = Array(I, verts, verts)

    # project onto subspace ⟂ 1
    if project
        P = I - ones(size(R))/size(R)[1]
        R = P*R*P
    end

    return R/tr(R)
end

""" Return the correlation matrix for
isotropic correlations with Preferred k Energy spectrum
E(k) = k exp(-k^2/(2 σ^2))

where C_{ij} = 2 ∫_0^∞ dk J_0(k r_{ij}) E(k) ∼ exp(-r_{ij}^2/(2 σ^2))
"""
function isotropic_gaussian_correlation_matrix(points::Array, σ::Float64; project=true)
    n_pts = size(points)[1]
    C = zeros(n_pts, n_pts)

    for i=1:size(points)[1]
        for j=i:size(points)[1]
            r_ij2 = norm(points[i,:] - points[j,:])^2
            # C[i,j] = 2σ*sqrt(π/2)*besselix(0, 0.25*r_ij2*σ^2)
            C[i,j] = exp(-0.5*r_ij2/σ^2)
        end
    end

    for i=1:size(points)[1]
        for j=1:i-1
            C[i,j] = C[j,i]
        end
    end

    # project onto subspace ⟂ 1
    if project
        P = I - ones(size(C))/size(C)[1]
        C = P*C*P
    end

    return C/tr(C)
end

""" Return the correlation matrix for
isotropic correlations with Gaussian Energy spectrum
E(k) = exp(-k^2/(2 σ^2))

where C_{ij} = 2 ∫_0^∞ dk J_0(k r_{ij}) E(k)
"""
function isotropic_gaussian_correlation_matrix(topol::NetworkTopology, σ::Float64; project=true)
    return isotropic_gaussian_correlation_matrix(topol.points, σ, project=project)
end

""" Return the correlation matrix for
isotropic correlations with energy spectrum

E(k) = exp(-k / a)

where C_{ij} = 2 ∫_0^∞ dk J_0(k r_{ij}) E(k)
"""
function isotropic_exponential_correlation_matrix(topol::NetworkTopology, a; project=true)
    return isotropic_exponential_correlation_matrix(topol.points, a, project=project)
end

""" Return the correlation matrix for
isotropic correlations with energy spectrum

E(k) = exp(-k / a)

where C_{ij} = 2 ∫_0^∞ dk J_0(k r_{ij}) E(k)
"""
function isotropic_exponential_correlation_matrix(points::Array, a::Float64; project=true)
    n_pts = size(points)[1]
    C = zeros(n_pts, n_pts)

    for i=1:size(points)[1]
        for j=i:size(points)[1]
            r_ij2 = norm(points[i,:] - points[j,:])^2
            C[i,j] = a./sqrt(1.0 + a^2*r_ij2)
        end
    end

    for i=1:size(points)[1]
        for j=1:i-1
            C[i,j] = C[j,i]
        end
    end

    # project onto subspace ⟂ 1
    if project
        P = I - ones(size(C))/size(C)[1]
        C = P*C*P
    end

    return C/tr(C)
end

end
