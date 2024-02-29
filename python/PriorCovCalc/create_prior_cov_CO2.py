#!/usr/bin/env python3

from datetime import date
import xarray as xr
import numpy as np
import os

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


def pick_area(lsm, latmin, latmax, lonmin, lonmax):
    lsm = lsm.where((latmin <= lsm.latitude) & (lsm.latitude <= latmax)
          &(lonmin <= lsm.longitude)& (lsm.longitude <= lonmax), drop=True) 
    lsm_flat = lsm.stack(latlon=("latitude", "longitude"))
    return lsm_flat
    

def compute_cov(lsm_flat, sigmas, L, prop_land = 0.5):

    nstate = lsm_flat.shape[0]
    #convert lsm to binary mask
    lsm_flat[lsm_flat <= prop_land] = 0
    lsm_flat[lsm_flat > prop_land] = 1
    #sort so that land first, then sea
    lsm_flat = lsm_flat.sortby(lsm_flat, ascending=False)
    #number of land gridcells = sum of land values
    nland = int(sum(lsm_flat.values))
    #initialize cov matrix
    cov = np.zeros((nstate, nstate))
    lat = lsm_flat["latitude"].values
    lon = lsm_flat["longitude"].values

    #compute covariances separately for land and ocean 
    for v in ["land", "ocean"]:
        print(v)
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
        coords={"lon": (["nparams"], lon),
                  "lat": (["nparams"], lat)},
        attrs={'comment': f"Prior covariance for CO2 bio fluxes"}
    )
    return out_cov



#land-sea-mask_0.1x0.1deg
PATHTOMASK = '/home/pietaril/Documents/data/masks/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc'
lsm = xr.open_dataset(PATHTOMASK)["lsm"][0]

#output 
OUTPUT_PATH = '/home/pietaril/Documents/data/unc_cov_matrices'
today = date.today()
output_filename = 'CO2_prior_cov_eur_%04d%02d.nc' % (today.year, today.month)

#Finland - approx  latitude: 59.6 - 70.1, longitude: 19.3 -31.6
#Europe: coords from Aki: -12, 39, 35, 74

latmin = -12.0
latmax = 39.0
lonmin = 35.0
lonmax = 74.0

lsm_flat = pick_area(lsm, latmin, latmax, lonmin, lonmax)

# Uncertainty (std)
sigmas = {'land': 0.8,
          'ocean': 1.2} 


# Correlation length (km)
L = {'land': 100,  # 
     'ocean': 500}  # for ocean

#compute cov matrix and coordinates ordered so that first land then ocean
cov, lat, lon = compute_cov(lsm_flat, sigmas, L)


# Output dataset
out_cov = create_dataset(cov, lat, lon)


out_cov.to_netcdf(os.path.join(OUTPUT_PATH, output_filename))

