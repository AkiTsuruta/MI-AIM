import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# Lukiessa datasettejä joutui pudottamaan kovarianssi- ja otoskovarianssimatriisit, koska tuo funktio ei jostain syystä halua
# antaa niille kolmatta dimensiota. Pitää lukea ne erikseen jotenkin.

np_t = [] # list for outputs with settings. invm = numpy.linalg.inv, useKG = True
np_f = [] # useKG = False
bl_t = [] # invm = block_iter_inv
bl_f = []

# block_iter_inv didn't work without Kalman filter with the first simulated dataset (probably, because
# there were only two states), so I'll only use data from simulations 1 to 5 (so not simulation 0)


for i in range(1,6):

    np_t.append(xr.open_mfdataset(f"simulated_data/simulation_0{i}/out_default_with_kf/*.nc",
                                    concat_dim='time2',combine='nested', drop_variables=["prior_cov", "posterior_cov", "obs_unc"]))

    bl_t.append(xr.open_mfdataset(f"simulated_data/simulation_0{i}/out_block_with_kf/*.nc",
                        concat_dim='time2',combine='nested', drop_variables=["prior_cov", "posterior_cov", "obs_unc"]))

    np_f.append(xr.open_mfdataset(f"simulated_data/simulation_0{i}/out_default_without_kf/*.nc",
                                    concat_dim='time2',combine='nested', drop_variables=["prior_cov", "posterior_cov", "obs_unc"]))
    
    bl_f.append(xr.open_mfdataset(f"simulated_data/simulation_0{i}/out_block_without_kf/*.nc",
                                    concat_dim='time2',combine='nested', drop_variables=["prior_cov", "posterior_cov", "obs_unc"]))





data = np_t[1]


fig, axs = plt.subplots(2,2, layout = 'constrained')
#plot posteriors for different states in state vector
axs[0,0].plot(data["posterior"][:,0], 'o')
axs[0,1].plot(data["posterior"][:,1], 'o')
axs[1,0].plot(data["posterior"][:,2], 'o')
axs[1,1].plot(data["posterior"][:,3], 'o')
plt.show()
















