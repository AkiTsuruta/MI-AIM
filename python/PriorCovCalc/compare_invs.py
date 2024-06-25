#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#datapath = "/home/pietaril/Documents/data/co2_prior_unc_cov/inv/"
#fname_py = "pyinv_test_land.nc"
#fname_kl = "testinv_land_8_10.nc"

datapath = "/scratch/project_462000459/maija/data/co2_prior_unc_cov/inv/"

rho = 2
n_neighbors = 9
v = "ocean"
fname_kl = f"KLinv_centeur_{v}_{rho}_{n_neighbors}.nc"
fname_py = f"pyinv_prior_co2_unc_centeur_{v}.nc"

ds_kl = xr.open_dataset(datapath+fname_kl)
iK_kl = ds_kl["inverse_cov"].values
ds_py = xr.open_dataset(datapath+fname_py)
iK_py = ds_py["inverse_cov"].values

diff = iK_py-iK_kl

#qs = np.quantile(diff.flatten(), [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
#print([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

#print(qs)


df = pd.DataFrame.from_dict({
    "frobenius" : [np.linalg.norm(diff)],
    "2-norm" : [np.linalg.norm(diff, ord=2)],
    "maxerr" : [abs(diff).max()],
    "rho"    : [rho],
    "n_neighbors" : [n_neighbors],
    "n" : [diff.shape[0]]})

df.to_csv(datapath+"stats/difference_to_pyinv.csv")



#plot the difference
diff = xr.DataArray(diff)
fig, ax = plt.subplots()
diff.plot.imshow(ax=ax, robust = True, origin='upper')
fig.suptitle(f"Difference matrix iK-iK_approx")
figname="diff_ocean_2_9"
plt.savefig(datapath+"fig/"+figname)
