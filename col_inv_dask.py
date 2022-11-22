#!/usr/bin/env python

import numpy as np
import xarray as xr
import scipy.sparse.linalg 
import dask
from dask import delayed
import dask.array as da
from functools import partial


invm = np.linalg.inv
gmres = scipy.sparse.linalg.gmres

#readdata here

M = bio_2.data.persist()
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