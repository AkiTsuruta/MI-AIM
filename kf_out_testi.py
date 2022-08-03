import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# Purkkaratkaisu matriisien lukemiseksi, kun xr-funktioilla ei näyttänyt
# fiksusti onnistuvan

B_s = [] # list for storing prior covariance matrices
A_s = [] # posterior covariance matrices
R_s = [] # observation error covariance matrices
for i in range(5):
    data = xr.open_dataset(f"simulated_data/simulation_00/s_00_out_{i:2d}.nc")
    B_s.append(data["prior_cov"])
    A_s.append(data["posterior_cov"])
    R_s.append(data["obs_unc"])


fig, axs = plt.subplots(2,3)
axs[-1,-1].axis('off')
for j in range(5):
    minval = A_s[j].values.min()
    maxval = A_s[j].values.max()
    if j < 3:
        A_s[j].plot.imshow(ax = axs[0,j], yincrease = False, robust = True)
    else:
        A_s[j].plot.imshow(ax=axs[1,j-3], yincrease = False, robust = True)
plt.tight_layout()
plt.show()

