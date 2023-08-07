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



# **********************#
# Compute factorization #
# **********************#

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


#************************************#
# Timing with different values for N #
#************************************#

times_KoLesky = Dict();
times_cho = Dict();



#Ns = [100, 1000, 5000, 10000]
#for N in Ns
    #x = rand(2, N)
    #K = kernel_matrix(x'[:,:],l,k)
    #times_KoLesky["$N"] = @elapsed compute_explicit_factor(x, K, rho, neighbors);
    #times_cho["$N"] = @elapsed cholesky(K)
#end



@time compute_explicit_factor(x, K, rho, neighbors);


using LinearAlgebra
 



# compute approximations of K and inv_K
#K_approx = KoLesky.assemble_covariance(factor)
#inv_K_approx = assemble_inv_covariance(factor)

#compare to exact

#inv_K = inv(K)



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




#need to zoom in to see anything

U_slice = LinearAlgebra.UpperTriangular(U[1:20, 1:20])
U_approx_slice = LinearAlgebra.UpperTriangular(U_approx[1:20, 1:20])

println(U_slice)
println(U_approx_slice)
#samples = sample_matrix_covariance(x, K, rho, neighbors, n_samples)

# Plot a sample
#n = 10
#scatter(x[1,:], x[2,:], marker_z = samples[:,n])





#**********#
# Plotting #
#**********#



#colormap = cgrad(:matter, 7, categorical=true);


using LinearAlgebra

#only select values in the upper triangular part of the matrix U for inspecting the distribution
# of values in that part and gather them to a vector

function select_uvals(A::AbstractArray)
    uvals = []
    n = size(A,2)
    for j in 1:n
        for i in 1:j
            push!(uvals, A[i,j])
        end
    end
    return uvals
end

U_uvals = select_uvals(U)
U_approx_uvals = select_uvals(U_approx)


#quantile(U_uvals, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
#quantile(U_approx_uvals, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])

#p1 = heatmap(U, yflip=true, c=:bwr, clims = (-14, 14), title = "Exact Cholesky factor U")
#lens!([1,50], [1,50], inset = (1, bbox(0.1, 0.45, 0.4, 0.4, )), yflip = true, clims = (-14, 14), legend = :none)



#p2 = heatmap(U_approx, yflip=true, c=:bwr, clims = (-14, 14), title = "Sparse approximation")
#lens!([1,50], [1,50], inset = (1, bbox(0.1, 0.45, 0.4, 0.4, )), yflip = true, clims = (-14, 14), legend = :none)

#p3 = heatmap(U.-U_approx, yflip=true, c=:bwr, clims = (-0.1, 0.1))





#spy(U_approx, c=:inferno, clims = (-14, 14))
#p1 = heatmap(U_slice, c = colormap, yflip = true, clims = (-100, 100))
#p2 = spy(U, clims = (-100, 100))
