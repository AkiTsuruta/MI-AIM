#!/usr/bin/env python
import xarray as xr
import numpy as np
import dask
import time
from scipy.linalg import solve


def colinv(M):
    n = len(M)
    cols = []
    exit_codes = []

    for i in np.arange(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        inv_col = dask.delayed(solve)(M, e_i, assume_a='sym')
        cols.append(inv_col)
    

    cols = dask.compute(*cols)
    iM = np.stack(cols, axis=1)

    return iM


M = xr.open_dataset("./data/regions_verify_202104_cov.nc", chunks = 'auto')["covariance_bio"].data[:100, :100].persist()

start = time.time()
iM = colinv(M)
end = time.time()
print(iM[0,:10])
print(f"Col inv with scipy.linalg.solve & dask.delayed, n={len(M)}, duration: {end-start} seconds")
