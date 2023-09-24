# Tests to see if I can implement KoLesky with our test covariance matrices instead of a covariance function
include("/home/pietaril/Documents/MI-AIM/julia/KoLesky.jl-master/src/KoLesky.jl")

using Random
using Plots
using StatsPlots
using Statistics
using StatsBase
using Distances
using NCDatasets
using LinearAlgebra


# read nc file & load matrix data
pathtodata = "/home/pietaril/Documents/data/CO2M_testdata/unc_cov_matrices/unc_cov_matrix20250101_nanmedian_small.nc";

ds = Dataset(pathtodata);
K = ds["covariance"][:,:];
lon = ds["lon"][:];
lat = ds["lat"][:];
close(ds);


x = [lat'; lon'];

a = max(0, -2eigmin(K)) * I

K = K + a


#*******************************************#
# Maija's helpers to interface with KoLesky #
#*******************************************#

function compute_explicit_factor(x, K, rho, neighbors)
    #returns ExplicitKLFactorization explicit_factor 
    measurements = KoLesky.point_index_measurements(x)
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    return explicit_factor
end    


#N = 6858 # how many locations
#x = rand(2,N) # random sampling


# length scale
l = [0.001]


# Control parameters for Florian's function
rho = 5 # accuracy of approximation. 2 = greedy, 8 = accurate but slow
neighbors = 9 # how many neighbors
n_samples = 100 # n samples from the spatial process


factor = compute_explicit_factor(x, K, rho, neighbors);


K_approx = KoLesky.assemble_covariance(factor)
K

