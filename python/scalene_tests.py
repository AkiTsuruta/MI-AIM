import numpy as np
invm = np.linalg.inv
import xarray as xr
from blockinv_iterative import block_iter_inv_fbm, block_iter_inv_fdm, block_iter_inv_fdi
from blockinv_iterative_symmetric import block_inv_s, block_iter_inv_s, block_iter_inv_s2
from scalene import scalene_profiler


ds_1 = xr.open_dataset("data/regions_verify_isotope_202112_cov.nc")
bio_1 = ds_1["covariance_bio"]
anth_1 = ds_1["covariance_anth"]

ds_2 = xr.open_dataset("data/regions_verify_202104_cov.nc")
bio_2 = ds_2["covariance_bio"]
anth_2 = ds_2["covariance_anth"]

M = bio_2.to_numpy()

inv = block_iter_inv_fdi(M)
