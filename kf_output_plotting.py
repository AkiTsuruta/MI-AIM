import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Tämä eka lukutapa jossa lisättiin uus dimensio oli Akin ehdotus virheen korjaamiseksi.  joutui kuitenkin pudottamaan kovarianssi ja otoskovarianssimatriisit, koska tuo funktio ei jostain syystä halua
# antaa niille kolmatta dimensiota. Pitää lukea ne erikseen jotenkin.

data = xr.open_mfdataset("output2/*.nc",concat_dim='time2',combine='nested', drop_variables=["prior_cov", "posterior_cov", "obs_unc"])
#data2 = xr.open_mfdataset("output/*.nc", concat_dim="tw", combine='nested', compat = 'override', drop_variables= ["prior", "posterior", "obs", "time", "differences"])
#print(data2)




# Oma yritys avata alla. Observationit tuli oikein, mutta tää hävitti priorin ja posteriorin kaikki muut paitsi ekat arvot
# ds = xr.open_mfdataset("output/*.nc", concat_dim= "time", combine="nested", data_vars='minimal', 
# coords='minimal', compat = "override", drop_variables=["prior_cov", "posterior_cov", "obs_unc"])



#En ole keksinyt miten nuo 5 aikaikkunan viivat saisi yhdistettyä
# fig, ax = plt.subplots(layout = 'constrained')
# ax.plot(data["time"], data["obs"].transpose(), color = 'blue', marker = 'o')
# plt.show()

# fig, axs = plt.subplots(2,2, layout = 'constrained')
# #plot posteriors for different states in state vector
# axs[0,0].plot(data["posterior"][:,0], 'o')
# axs[0,1].plot(data["posterior"][:,1], 'o')
# axs[1,0].plot(data["posterior"][:,2], 'o')
# axs[1,1].plot(data["posterior"][:,3], 'o')
# plt.show()



print(data['differences'].values)

# Alla piirrettynä obs - H*y erot kussakin aikaikkunassa.
# Koska ensimmäisessä aikaikkunassa erot niin suuret, 
# niin voisi vielä erikseen piirtää muut aikaikkunat
# ilman ensimmäistä (tai tutkailla voiko y-akselin "katkaista")

fig, ax = plt.subplots(layout = 'constrained')
ax.plot(data['time'], data['differences'].transpose())
ax.set_ylabel('Difference')
ax.set_xlabel('Time')
ax.set_title('Obs - prior differences')
plt.show()
















