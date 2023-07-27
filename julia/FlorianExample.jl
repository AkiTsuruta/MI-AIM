# cd /Users/ottol/Documents/Julia/KoLesky.jl-master
#push!(LOAD_PATH,"/Users/ottol/Documents/Julia/KoLesky.jl-master/src/")
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


#Looks like the function below is not used at all in this example
# Function for computing kernel matrix for training data 
function kernel_function(X::Array{T}, l::Vector{T}, k::Function) where T # short for `where T<:Any`
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
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    samples = KoLesky.sample(explicit_factor, n_samples)
    # horizontally concatenates all the sample vectors into a single matrix
    samples = hcat(samples...)
    return samples
end

# *************** #
# Test functions  #
# *************** #

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
rho = 5 # accuracy of approximation. 2 = greedy, 8 = accurate but slow
neighbors = 3 # how many neighbors
n_samples = 100 # n samples from the spatial process

samples = sample_matrix_covariance(x, K, rho, neighbors, n_samples)

# Plot a sample
n = 10
scatter(x[1,:], x[2,:], marker_z = samples[:,n])
