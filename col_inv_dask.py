#!/usr/bin/env python

import numpy as np
import xarray as xr
from scipy.sparse.linalg import gmres
import dask


def write_to_file(M, iM, exit_codes, filename):
    out = xr.Dataset()
    out["cov"] = (('nstate'), ('nstate'), M)
    out["invcov"] = (('nstate'), ('nstate'), iM)
    out["exit_codes"] = exit_codes
    out.to_netcdf(f"outputs/{filename}")

def colinv(M, tol = 1e-8):
    n = len(M)
    cols = []
    exit_codes = []

    for i in np.arange(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        inv_col = dask.delayed(gmres)(M, e_i, tol = 1e-8)
        cols.append(inv_col[0])
        exit_codes.append(inv_col[1])

    cols = dask.compute(*cols)
    exit_codes = dask.compute(*exit_codes)
    iM = np.stack(cols, axis=1)

    return iM, exit_codes


ds_2 = xr.open_dataset("data/regions_verify_202104_cov.nc", chunks = 'auto')
bio_2 = ds_2["covariance_bio"]
M = bio_2.data.persist()

iM, exit_codes = colinv(M)
write_to_file(M, iM, exit_codes, "test1")




