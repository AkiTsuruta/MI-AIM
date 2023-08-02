include("/home/pietaril/Documents/MI-AIM/julia/KoLesky.jl-master/src/KoLesky.jl")

using Random
using Plots
using StatsPlots
using Statistics
using StatsBase
using Distances


# ************************************** #
# Functions for example kernel modeling  #
# ************************************** #

# Square exponential kernel
function spherical_sqexp(d::T; logθ::AbstractVector{T}) where T <: Real
    exp(logθ[1]) * exp(-.5d^2 / (exp(logθ[2]) .^ 2))
end

# Matern 3/2 kernel
function Matern32(d::T; logθ::AbstractVector{T}) where T <: Real
    h = sqrt(3)*d / exp(logθ[2]) # d is Euclidean distance
    exp(logθ[1]) * (1. + h) * exp(-h)
end

# Function for computing kernel matrix for training data
function kernel_function(X::Array{T}, l::Vector{T}, k::Function) where T 
    # Square exponential
    # pairwise(Euclidean) doesn't work with Zygote due to mutating arrays
    #d = pairwise(Euclidean(),eachcol((@view X[1,:])), eachrow((@view X[2:end,:])))
    d = sqrt.(sum((((@view X[1,:])' .- (@view X[2:end, :]))).^2, dims=2))
    k.(d; logθ = [0.0; l])
end

# Function for computing kernel of new location vs training data
function kernel_matrix(X::Array{T}, l::Vector{T},  k::Function) where T
    d = pairwise(Euclidean(), X')
    k.(d; logθ = [0.0; l])
end

# ****************************************** #
# Helper function to interface with KoLesky  #
# ****************************************** #

function sample_matrix_covariance(x, K, rho, neighbors, n_samples)
    measurements = KoLesky.point_index_measurements(x)
    #Turns K into a MatrixCovariance object (i.e. not simply an array). MatrixCovariance has the method cov_mat 
    #that returns the covariance in matrix form. So K = G.cov_mat
    G = KoLesky.MatrixCovariance(K) 
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    samples = KoLesky.sample(explicit_factor, n_samples)
    # horizontally concatenates all the sample vectors into a single matrix
    samples = hcat(samples...)
    return samples
    
end

#*****************#
# Maija's helpers #
#*****************#

function compute_explicit_factor(x, K, rho, neighbors)
    #returns ExplicitKLFactorization explicit_factor 
    measurements = KoLesky.point_index_measurements(x)
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    return explicit_factor
end    
    

function assemble_inv_covariance(factor::KoLesky.ExplicitKLFactorization)
    #returns approximation of the inverse of K
    U_matrix = Matrix(factor.U)
    inv_P = similar(factor.P)
    inv_P[factor.P] = 1 : length(inv_P)

    return (U_matrix * U_matrix')[inv_P, inv_P]
end 



# *******#
# Tests  #
# *******#

# Example random locations
N = 10000 # how many locations
x = rand(2,N) # random sampling

# Construct example spatial covariance matrix
# select kernel
#k = spherical_sqexp
k = Matern32
# length scale
l = [0.001]
# kernel matrix
K = kernel_matrix(x'[:,:],l,k)

# Control parameters for Florian's function
rho = 8 # accuracy of approximation. 2 = greedy, 8 = accurate but slow
neighbors = 10 # how many neighbors
n_samples = 100 # n samples from the spatial process


factor = compute_explicit_factor(x, K, rho, neighbors);

 
# compute approximations of K and inv_K
#K_approx = KoLesky.assemble_covariance(factor)
#inv_K_approx = assemble_inv_covariance(factor)

#compare to exact

#inv_K = inv(K)


import LinearAlgebra
#relative error
#println(LinearAlgebra.norm(inv_K .- inv_K_approx)/LinearAlgebra.norm(inv_K))
#println(LinearAlgebra.norm(K .- K_approx)/LinearAlgebra.norm(K))

#check if inv_K_approx * K ≈ I
#inv_K_approx * K


# Largeish relative error between inv_K and inv_K_approx. 
# According to Otto this is probably due to U being a sparse approximation
# Compare instead U_approx to actual Cholesky U
U_approx = Matrix(factor.U)
C = LinearAlgebra.cholesky(K[factor.P, factor.P])
U = inv(C.U)



#samples = sample_matrix_covariance(x, K, rho, neighbors, n_samples)

# Plot a sample
#n = 10
#scatter(x[1,:], x[2,:], marker_z = samples[:,n])



#plot
#values in the array are from such a large scale that
#it looks like it's mainly zeros. Need to adjust the plot
#ranges to be able to see something
#heatmap(denseCov)

