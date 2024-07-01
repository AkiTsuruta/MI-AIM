#!/usr/bin/julia
module KLHelpers
#include("/home/pietaril/Documents/MI-AIM/julia/KoLesky.jl-master/src/KoLesky.jl")
#in LUMI
include("/project/project_462000459/maija/kl_inv/KoLesky.jl-master/src/KoLesky.jl")

using LinearAlgebra
using NCDatasets
using JLD
using Distances


function expker(X::Array{<:Real,2}, sigma::Real, l::Integer)
    # Inputs: 2xN-matrix X with longitudes as first row, latitudes as second
    d = pairwise(Haversine(), X)./1000; #in km
    @.d = sigma^2*exp.(-d./l);

end

function compute_explicit_factor(K::Matrix{T}, x::Matrix{T}, rho::Int64, n_neighbors::Int64) where T <: Real
    #returns ExplicitKLFactorization explicit_factor 
    measurements = KoLesky.point_index_measurements(x)
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, n_neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
    #return explicit_factor.U, explicit_factor.P
    return explicit_factor
end    


function assemble_inv_covariance(factor)
    #returns approximation of the inverse of K
    U_matrix = Matrix(factor.U)
    inv_P = similar(factor.P)
    inv_P[factor.P] = 1 : length(inv_P)

    return (U_matrix * U_matrix')[inv_P, inv_P]
end 


function make_PD!(K::Matrix{T}, a::T) where T
    # Make input matrix positive definite by adding the identity matrix I multiplied by a small constant. 
    K[:,:] = K + a*I;

    return K
end


function read_cov_from_file(filepath::String)
    ds = Dataset(filepath);
    K = ds["covariance"][:,:];
    #julia thinks K still contains missing values, convert the type
    K = convert(Matrix{typeof(K[1,1])}, K)
    lon = ds["lon"][:];
    lat = ds["lat"][:];
    close(ds);
    x = [lon'; lat'];
    x = convert(Matrix{typeof(x[1,1])}, x);

    return K, x
end


function read_iK_from_file(filepath::String)
    ds = Dataset(filepath);
    iK = ds["inverse_cov"][:,:];
    close(ds);
    iK = convert(Matrix{typeof(iK[1,1])}, iK);

    return iK
end

function read_coords_from_file(filepath::String, useFloat32::Bool)
    ds = Dataset(filepath);
    lon = ds["lon"][:];
    lat = ds["lat"][:];
    close(ds);
    x = [lon'; lat'];
    if useFloat32
        x = convert(Matrix{Float32}, x);
    else
        x = convert(Matrix{Float64}, x);
    end

    return x
end


function KL_invert(K::Matrix{T}, x::Matrix{T}, rho::Integer, n_neighbors::Integer) where T <: Real
    factor = compute_explicit_factor(K, x, rho, n_neighbors)
    iK_approx = assemble_inv_covariance(factor);
    
    return iK_approx
end




function write_factor_to_file(factor, rho::Integer, n_neighbors::Integer, outpath::String)
    #For saving U and P and parameters n, rho, n_neighbors. Filetype here
    # is jld because couldn't save sparse matrices in netcdf
    U = factor.U;
    P = factor.P;
    n = size(U, 1);
    
    file = jldopen(outpath, "w", compress=true);
    write(file, "U", U);
    write(file, "P", P);
    write(file, "n", n);
    write(file, "rho", rho);
    write(file, "n_neighbors", n_neighbors);
    close(file);
    
end

function write_iK_to_file(iK_approx::Matrix{T}, x::Matrix{T}, outpath::String, rho::Integer, n_neighbors::Integer) where T <: Real
    #For saving the entire inverted cov matrix as a netcdf file
    n = size(iK_approx, 1);
    out = NCDataset(outpath, "c");
    defDim(out, "nstate", n);
    v = defVar(out, "inverse_cov", typeof(iK[1,1]), ("nstate", "nstate"));
    v[:,:] = iK_approx;
    v.attrib["rho"] = rho;
    v.attrib["n_neighbors"] = n_neighbors;
    v.attrib["n"] = n;
    v.attrib["comments"] = "Approximate inverse of covariance matrix calculated with KoLesky";
    #add variable for coordinates
    lon = defVar(out, "lon", typeof(x[1,1]), ("nstate",));
    lon[:] = x[1, :];
    lat = defVar(out, "lat", typeof(x[2,1]), ("nstate",));
    lat[:] = x[2, :];
    #v.attrib["time"] = time;
    close(out)
    
end 


function main(filepath::String, rho::Integer, n_neighbors::Integer, a=1e-16)
    # read in covariance matrix K and lonlat -coordinates x
    K, x = read_cov_from_file(filepath);
    #ensure K positive definite
    K = make_PD!(K, a);
    #factorize
    factor = compute_explicit_factor(x, K, rho, n_neighbors);
    date = basename(filepath)[15:22];
    n = size(K)[1];
    filename = "KL_Factorization$(date)_n$n";
    write_factor_to_file(factor, rho, n_neighbors, filename);

end




#filepath = "/home/pietaril/Documents/data/CO2M_testdata/unc_cov_matrices/unc_cov_matrix20250609_nanmedian_small.nc";
#main(filepath, 2, 9)

end