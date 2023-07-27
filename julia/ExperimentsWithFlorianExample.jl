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


#*********************************************#
# Maija's helper to get the U of Cholesky out #
#*********************************************#

function inv_cholesky(x, K, rho, neighbors)
    measurements = KoLesky.point_index_measurements(x)
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    return explicit_factor.U #this should be the U from the approx Cholesky factorization of Cov^(-1)
end    
    
#But is the covariance matrix now K or G? Answer: It is K. G is a MatrixCovariance object that is a covariance function


# *************** #
# Test functions  #
# *************** #

# Example random locations
N = 1000 # how many locations
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
rho = 5 # accuracy of approximation. 2 = greedy, 8 = accurate but slow
neighbors = 3 # how many neighbors
n_samples = 100 # n samples from the spatial process

samples = sample_matrix_covariance(x, K, rho, neighbors, n_samples)

# Plot a sample
#n = 10
#scatter(x[1,:], x[2,:], marker_z = samples[:,n])

#sparse array U
U = inv_cholesky(x, K, rho, neighbors)

# now Cov^(-1) = U'U so it should be Cov = U^(-1)*U^(-1)'

#convert to dense

denseU = Array(U)

idenseU = inv(denseU)
denseCov = idenseU*idenseU'


#plot
#values in the array are from such a large scale that
#it looks like it's mainly zeros. Need to adjust the plot
#ranges to be able to see something
#heatmap(denseCov)

