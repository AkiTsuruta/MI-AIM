#!/usr/bin/julia

include("/home/pietaril/Documents/MI-AIM/julia/KLHelpers.jl")
#include("/project/project_462000459/maija/kl_inv/KLHelpers.jl")

using NCDatasets
#using Distances



function write_cov_to_file(K::Matrix{T}, x::Array{T,2}, v::String, outpath::String, filename::String) where T <: Real
    #For saving the cov matrix and coordinates as a netcdf file
    n = size(K, 1);

    out = NCDataset(outpath*filename, "c");
    defDim(out, "nstate", n);
    variab = defVar(out, "covariance", typeof(K[1,1]), ("nstate", "nstate"));
    variab[:,:] = K;
    variab.attrib["land_or_ocean"] = v;
    lat = defVar(out, "lat", typeof(x[2,1]), ("nstate",));
    lon = defVar(out, "lon", typeof(x[1,1]), ("nstate",));
    lat[:] = x[2,:];
    lon[:] = x[1,:];
    close(out);
    
end 


function main(v::String, datapath::String, area::String, useFloat32::Bool)
    println("Reading in coordinates");
    x = KLHelpers.read_coords_from_file(datapath*"coords/$(area)_$(v)_coords.nc", useFloat32);

    # Uncertainty (std)
    sigmas = Dict([("land", 0.8),("ocean", 1.2)]);


    # Correlation length (km)
    L = Dict([("land", 100), ("ocean", 500)]);

    println("Computing cov matrix")
    K = KLHelpers.expker(x, sigmas[v], L[v]);

    fname= "prior_co2_unc_$(area)_$(v).nc";
    println("Writing to file")
    write_cov_to_file(K, x, fname, v, datapath);
    println("Done")

end


#datapath="/scratch/project_462000459/maija/data/co2_prior_unc_cov/";
datapath = "/home/pietaril/Documents/data/co2_prior_unc_cov/";
#main("ocean", datapath, "wholeur", true)




v = "land";
#filepath = "/scratch/project_462000459/maija/data/coords/$(v)_coords.nc";
pathtocoords = "/home/pietaril/Documents/data/co2_prior_unc_cov/coords/centeur_$(v)_coords.nc";



#outpath = "/home/pietaril/Documents/data/co2_prior_unc_cov"
#filename="memtest1";

#@time write_cov_to_file(K, x, v, outpath, filename);

