include("/home/pietaril/Documents/MI-AIM/julia/KoLesky.jl-master/src/KoLesky.jl")

using LinearAlgebra
using NCDatasets


function compute_explicit_factor(x::Matrix{T}, K::Matrix{T}, rho::Int64, neighbors::Int64) where T <: Real
    #returns ExplicitKLFactorization explicit_factor 
    measurements = KoLesky.point_index_measurements(x)
    G = KoLesky.MatrixCovariance(K)
    implicit_factor = KoLesky.ImplicitKLFactorization(G, measurements, rho, neighbors)
    explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor)
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
    K = convert(Array{Float64}, K)
    lon = ds["lon"][:];
    lat = ds["lat"][:];
    close(ds);
    x = [lat'; lon'];
    x = convert(Array{Float64}, x)

    return K, x
end

function KL_invert(x::Matrix{T}, K::Matrix{T}, rho::Int64, neighbors::Int64) where T <: Real
    factor = compute_explicit_factor(x, K, rho, neighbors)
    iK_approx = assemble_inv_covariance(factor);
    
    return iK_approx
end

function write_iK_to_file(iK_approx::Matrix{T}, filename::String, rho::Int64, neighbors::Int64) where T <: Real
    #For saving the entire inverted cov matrix as a netcdf file
    n = size(iK_approx, 1);
    out = NCDataset("/home/pietaril/Documents/data/outputs/$filename.nc", "c");
    defDim(out, "nstate", n);
    v = defVar(out, "inverse_cov_approx", Float64, ("nstate", "nstate"));
    v[:,:] = iK_approx;
    v.attrib["rho"] = rho;
    v.attrib["n_neighbors"] = neighbors;
    v.attrib["n"] = n;
    close(out)
    
end 

function main(datapath::String, rho::Int64, neighbors::Int64, a = 1e-16)
    K, x = read_cov_from_file(datapath);
    make_PD!(K,a);
    iK_approx = KL_invert(x, K, rho, neighbors);
    write_iK_to_file(iK_approx, "iK_test1", rho, neighbors);
    println("done");
end


datapath = "/home/pietaril/Documents/data/CO2M_testdata/unc_cov_matrices/unc_cov_matrix20250609_nanmedian_small.nc";

rho = 8;
neighbors = 9;

main(datapath, rho, neighbors);




