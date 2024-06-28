#!/usr/bin/julia
include("/home/pietaril/Documents/MI-AIM/julia/KLHelpers.jl")

using LinearAlgebra
using DataFrames
using Plots
using CSV
using Statistics

v = "ocean";
rho = 2;
n_neighbors = 9;

datapath = "/home/pietaril/Documents/data/co2_prior_unc_cov/inv/";
fname_py = "pyinv_test_land.nc";
fname_kl = "testinv_land_8_10.nc";



function compare_invs(vs::AbstractVector{String}, rhos::AbstractVector{<:Integer}, neighborss::AbstractVector{<:Integer}) 
    # Tests for timing Cholesky of Matern32 with l = [0.001]. Need to adjust this later to take also k and l as inputs so that they can also vary.
    # For some reason passing a function k didn't work
    res = DataFrame()
    i = 0
    for v in vs
        iK_py = KLHelpers.read_iK_from_file(datapath*"pyinv_test_land.nc");
        #pre-allocate diffr matrix
        diffr = zeros(size(iK_py));
        for rho in rhos
            for neighbors in neighborss
                    i += 1
                    #read kl-inverted matrix into the difference matrix
                    diffr .= KLHelpers.read_iK_from_file(datapath*"KLinv_centeur_$(v)_$(rho)_$(neighbors).nc");

                    # in-place subtraction
                    diffr .= iK_py .- diffr;

                    frob = norm(diffr);
                    frob_rel = frob / norm(iK_py);
                    maxerr = maximum(abs, diffr);
                    qs = quantile(diffr[:], [0.1, 0.25, 0.5, 0.75, 0.9])

                    
                    push!(res, (id = i, N = size(diffr)[1], v = v, rho = rho, n_neighbors = neighbors,
                        frobenius = frob, relative_frobenius = frob_rel, maxerr=maxerr, quantiles=qs))
                
            end
        end
    end
    return res

end

res = compare_invs(["land"], [2], [9])

norm(diffr)
maximum(abs, diffr)



#the colorbar doesn't center at zero for some reason
#colormap = cgrad(:vik);
#p2 = heatmap(difference, c = colormap, yflip = true)

