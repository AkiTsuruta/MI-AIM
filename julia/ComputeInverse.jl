#!/usr/bin/julia

#in LUMI
#include("/project/project_462000459/maija/kl_inv/KoLesky.jl-master/src/KoLesky.jl")
#include("/project/project_462000459/maija/kl_inv/KLHelpers.jl")

include("/home/pietaril/Documents/MI-AIM/julia/KoLesky.jl-master/src/KoLesky.jl")
include("/home/pietaril/Documents/MI-AIM/julia/KLHelpers.jl")


using DataFrames
using LinearAlgebra
using NCDatasets
using SparseArrays
using JLD




#pathtocov = "/home/pietaril/Documents/data/co2_prior_unc_cov/testcov_$v.nc";
#outpath_factor = "/home/pietaril/Documents/data/co2_prior_unc_cov/factor/testfactor_$v.jld";
#outpath_iK = "/home/pietaril/Documents/data/co2_prior_unc_cov/inv/testinv_$(v)_$(rho)_$(n_neighbors).nc";
for v in ["land", "ocean"]
    datapath = "/home/pietaril/Documents/data/co2_prior_unc_cov/"
    #datapath = "/scratch/project_462000459/maija/data/co2_prior_unc_cov/";
    covname = "testcov_$v.nc"
    #covname = "prior_co2_unc_centeur_$v.nc";
    println("reading matrix $covname")
    K, x = KLHelpers.read_cov_from_file(datapath*covname);
    for rho in [2]
        for n_neighbors in [9, 16]
            outpath_factor = datapath*"factor/KLfactor_centeur_$(v)_$(rho)_$(n_neighbors).jld";
            println("rho: $(rho), n_neighbors: $(n_neighbors), v: $(v)")
            println("Computing explicit factor")
            factor = KLHelpers.compute_explicit_factor(K, x, rho, n_neighbors);
            println("Writing factor to file")
            KLHelpers.write_factor_to_file(factor, rho, n_neighbors, outpath_factor);
            outpath_iK = datapath*"inv/KLinv_centeur_$(v)_$(rho)_$(n_neighbors).nc";
            println("Assembling inverse")
            iK = KLHelpers.assemble_inv_covariance(factor);
            println("Writing inverse to file")
            KLHelpers.write_iK_to_file(iK, x, outpath_iK, rho, n_neighbors);
        end
    end
end

println("Finished")






