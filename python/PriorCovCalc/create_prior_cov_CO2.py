#!/usr/bin/env python3

from datetime import date
import xarray as xr
import numpy as np
from global_land_mask import globe
import matplotlib.pyplot as plt

def haversine_distance(lat, lon):
    """
    Calculate pairwise distances between points using Haversine formula.

    Parameters:
    - lat (ndarray) : 1-d array of latitude coordinates
    - lon (ndarray) : 1-d array of longitude coordinates
    Returns:
    - distances (ndarray): Pairwise distances matrix.
    """

    # Earth radius in kilometers
    earth_radius = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Compute differences in latitude and longitude
    dlat = lat_rad[:, np.newaxis] - lat_rad
    dlon = lon_rad[:, np.newaxis] - lon_rad

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat_rad[:, np.newaxis]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate pairwise distances in kilometers
    distances = earth_radius * c

    return distances

def create_lsm(latmin, latmax, lonmin, lonmax):
     """Creates a land-sea mask (land=1, sea=0) in 0.1 deg x 0.1 deg resolution
     for the chosen area.
     """
     lat = np.linspace(latmin,latmax, (latmax-latmin)*10)
     lon = np.linspace(lonmin,lonmax, (lonmax - lonmin)*10)
     lon_grid, lat_grid = np.meshgrid(lon,lat)
     z = globe.is_land(lat_grid, lon_grid)
     z_int = np.array(z, dtype=int)
     lsm = xr.DataArray(z_int, coords=[lat, lon], dims=["lat", "lon"])
     return lsm
   

def compute_cov(lsm, sigmas, L):
    """The function computes a spatial covariance matrix using
    formula sigma^2*exp(-d/l), where d is the distance between
    cells, l is the length-scale and sigma is the standard deviation
    """

    lsm_flat = lsm.stack(latlon=("lat", "lon"))

    nstate = lsm_flat.shape[0]

    #sort so that land cells first, then sea
    lsm_flat = lsm_flat.sortby(lsm_flat, ascending=False)
    #number of land gridcells = sum of land values
    nland = int(sum(lsm_flat.values))
    print(f"number of land cells: {nland}")
    #initialize cov matrix
    cov = np.zeros((nstate, nstate))
    lat = lsm_flat["lat"].values
    lon = lsm_flat["lon"].values

    #compute covariances separately for land and ocean 
    for v in ["land", "ocean"]:
        if v == "land":
            inds = [0, nland]
       
        else:
            inds = [nland, nstate]

        latv = lat[inds[0]:inds[1]]
        lonv = lon[inds[0]:inds[1]]  
        dists = haversine_distance(latv, lonv)
        sigma = sigmas[v]
        l = L[v]
        cov[inds[0]:inds[1], inds[0]:inds[1]] = sigma**2*np.exp(-1*(dists/l))
        
    return cov, lat, lon
    

def create_dataset(covariance, lat, lon):
    
    """
    Creates an xarray dataset of the covariance matrix and coordinates of the grid.
    """
    out_cov = xr.Dataset(
        data_vars={"covariance": (["nparams", "nparams"], covariance)},
        coords={"lon": (["nparams"], np.asarray(lon).flatten()),
                  "lat": (["nparams"], np.asarray(lat).flatten()),},
        attrs={'comment': f"Prior covariance for CO2 bio fluxes"}
    )
    return out_cov







#output 
OUTPUT_PATH = '/home/pietaril/Documents/data/unc_cov_matrices'
today = date.today()
output_filename = 'CO2_prior_cov_eur_%04d%02d.nc' % (today.year, today.month)


latmin = 30
latmax = 40
lonmin = -10
lonmax = 0

# latmin = 30
# latmax = 75
# lonmin = -15
# lonmax = 40

lsm = create_lsm(latmin, latmax, lonmin, lonmax)

# Uncertainty (std)
sigmas = {'land': 0.8,
          'ocean': 1.2} 


# Correlation length (km)
L = {'land': 100,  # 
     'ocean': 500}  # for ocean

#compute cov matrix and coordinates ordered so that first land then ocean
cov, lat, lon = compute_cov(lsm, sigmas, L)
print(cov.shape)

plt.figure(1)
plt.title("Prior cov xco2")
plt.pcolormesh(cov)
plt.colorbar()
plt.savefig()

# Output dataset
#out_cov = create_dataset(cov, lat, lon)


#out_cov.to_netcdf(os.path.join(OUTPUT_PATH, output_filename))

