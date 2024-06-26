#!/usr/bin/julia

include("/home/pietaril/Documents/MI-AIM/julia/KLHelpers.jl")
#include("/project/project_462000459/maija/kl_inv/KLHelpers.jl")

using NCDatasets
using Distances


function write_cov_to_file(K::Matrix{T}, x::Array{T,2}, v::String, outpath::String, filename::String) where T <: Real
    #For saving the cov matrix and coordinates as a netcdf file
    n = size(K, 1);

    out = NCDataset("$(outpath)/$(filename).nc", "c");
    defDim(out, "nstate", n);
    variab = defVar(out, "covariance", Float64, ("nstate", "nstate"));
    variab[:,:] = K;
    variab.attrib["land_or_ocean"] = v;
    lat = defVar(out, "lat", Float64, ("nstate",));
    lon = defVar(out, "lon", Float64, ("nstate",));
    lat[:] = x[2,:];
    lon[:] = x[1,:];
    close(out);
    
end 





v = "land";
#filepath = "/scratch/project_462000459/maija/data/coords/$(v)_coords.nc";
pathtocoords = "/home/pietaril/Documents/data/co2_prior_unc_cov/coords/centeur_$(v)_coords.nc";


x = KLHelpers.read_coords_from_file(pathtocoords);
x = x[:,1:100];



# Uncertainty (std)
sigmas = Dict([("land", 0.8),("ocean", 1.2)]); 


# Correlation length (km)
L = Dict([("land", 100), ("ocean", 500)]);

K = KLHelpers.expker(x, sigmas[v], L[v]);

outpath = "/home/pietaril/Documents/data/co2_prior_unc_cov"
filename="testcov_$v";

#write_cov_to_file(K, x, v, outpath, filename);

