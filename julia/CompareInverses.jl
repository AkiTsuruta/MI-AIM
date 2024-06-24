#!/usr/bin/julia
include("/home/pietaril/Documents/MI-AIM/julia/KLHelpers.jl")

using LinearAlgebra
using NCDatasets
using Plots

datapath = "/home/pietaril/Documents/data/co2_prior_unc_cov/inv/";
fname_py = "pyinv_test_land.nc";
fname_kl = "testinv_land_8_10.nc";

iK_kl = KLHelpers.read_iK_from_file(datapath*fname_kl)
iK_py = KLHelpers.read_iK_from_file(datapath*fname_py)
difference = iK_py - iK_kl
norm(difference)
opnorm(difference)
maximum(difference)

colormap = cgrad(:vik);
p2 = heatmap(difference, c = colormap, yflip = true)

