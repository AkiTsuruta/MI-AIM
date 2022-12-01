#!/usr/bin/env python
import xarray as xr
import numpy as np
import multiprocessing as mp
from scipy.sparse.linalg import gmres
import time




def write_to_file(M, iM, exit_codes, filename):
    out = xr.Dataset()
    out["cov"] = (('nstate', 'nstate'), M)
    out["invcov"] = (('nstate','nstate'), iM)
    out["exit_codes"] = ('nstate', list(exit_codes))
    out.to_netcdf(f"outputs/{filename}.nc")
    


def parallel_gmres(M, tol=1e-8):
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    n = len(M)
    #this might be possible to be done with only creating each std unit vector
    eye = np.eye(n)
    icols_exitvals = pool.starmap(gmres, [(M, col, None, tol) for col in eye])

    return icols_exitvals


def main(filepath):
    M = xr.open_dataset(filepath)["covariance_bio"].values
    start = time.time()
    icols_exitvals = parallel_gmres(M)
    #split inverted columns and exits to two tuples
    icols, exits = zip(*icols_exitvals)
    #stack the columns into matrix form
    iM = np.stack(icols, axis = 1)
    end = time.time()
    print(iM[0,:])
    print(f"Inversion completed in {end-start} seconds")
    #write_to_file(M, iM, exits, "testinv1")

main("data/regions_verify_isotope_202112_cov.nc")
