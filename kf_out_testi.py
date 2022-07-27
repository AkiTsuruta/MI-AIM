import xarray as xr
import numpy as np
import matplotlib.pyplot as plt



# Purkkaratkaisu matriisien lukemiseksi, kun xr-funktioilla ei näyttänyt
# fiksusti onnistuvan

Bs = [] # list for storing prior covariance matrices
As = [] # posterior covariance matrices
Rs = [] # observation error covariance matrices
for i in range(5):
    data = xr.open_dataset(f"kf_outputs/output3/out_0{i}.nc")
    Bs.append(data["prior_cov"])
    As.append(data["posterior_cov"])
    Rs.append(data["obs_unc"])


for i in range(1):
    fig, ax = plt.subplots(layout = 'constrained')
    As[i].plot.pcolormesh(yincrease = False, robust = True)
    plt.show()


