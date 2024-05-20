#!/usr/bin/env python3

from datetime import datetime
import xarray as xr
import numpy as np
from global_land_mask import globe
import matplotlib.pyplot as plt
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


def create_lsm(latmin, latmax, lonmin, lonmax):
     """Creates a land-sea mask (land=1, sea=0) in 0.1 deg x 0.1 deg resolution
     for the chosen area.
     """
     lat = np.linspace(latmin,latmax, (latmax-latmin)*10+1, dtype=np.float32)
     lon = np.linspace(lonmin,lonmax, (lonmax - lonmin)*10+1, dtype = np.float32)
     longrid, latgrid = np.meshgrid(lon,lat)
     lsm = globe.is_land(latgrid, longrid)
   
     return lsm, latgrid, longrid


    
def compute_cov(lsm, latgrid, longrid, sigmas, L):
    """The function computes spatial covariance matrices for land and ocean areas using
    formula sigma^2*exp(-d/l), where d is the distance between
    cells, l is the length-scale and sigma is the standard deviation.

    Returns two dictionaries: cov and coords 
    """

    cov = {}
    coords = {}
       
    #compute covariances separately for land and ocean 
    for v in ["land", "ocean"]:
        print(f"computing {v} coordinates")
        if v == "land":
             latv = latgrid[lsm].flatten()
             lonv = longrid[lsm].flatten()
             
            
        else:
            latv = latgrid[~lsm].flatten()
            lonv = longrid[~lsm].flatten()
             
        sigma = sigmas[v]
        l = L[v]
        print(f"computing {v} cov")
        covv = sigma**2*np.exp(-1*haversine_distance(latv, lonv)/l)
        cov[v] = covv
        coords[v] = {"lon": lonv, "lat":latv}
        
    return cov, coords

    

def create_dataset(covariance, lat, lon, v):
    
    """
    Creates an xarray dataset of the covariance matrix and coordinates of the grid.
    """
    out_cov = xr.Dataset(
        data_vars={"covariance": (["nparams", "nparams"], covariance)},
        coords={"lon": (["nparams"], np.asarray(lon).flatten()),
                  "lat": (["nparams"], np.asarray(lat).flatten()),},
        attrs={'comment': f"Prior covariance matrix of CO2 {v} fluxes"}
    )
    return out_cov





#output 
#OUTPUT_PATH = '/home/pietaril/Documents/data/unc_cov_matrices'
OUTPUT_PATH = '/scratch/project_462000459/maija/data/co2_prior_unc_cov'

today = datetime.now()

# latmin = 30
# latmax = 40
# lonmin = -10
# lonmax = 0

latmin = 30
latmax = 75
lonmin = -15
lonmax = 40

lsm, latgrid, longrid = create_lsm(latmin, latmax, lonmin, lonmax)

# Uncertainty (std)
sigmas = {'land': 0.8,
          'ocean': 1.2} 


# Correlation length (km)
L = {'land': 100,  # 
     'ocean': 500}  # for ocean

#compute land and ocean cov matrices separately

covs, coords = compute_cov(lsm, latgrid, longrid, sigmas, L)

for v in ["land", "ocean"]:
    print(f"creating {v} output file")
    #ii = 1
    output_filename = f'CO2_prior_cov_eur_{v}_{today.strftime("%d%m%Y")}.nc'
    out_cov = create_dataset(covs[v], coords[v]["lat"], coords[v]["lon"], v)
    out_cov.to_netcdf(os.path.join(OUTPUT_PATH, output_filename))

    # plt.figure(ii)
    # plt.title(f"Prior cov xco2 {v}")
    # plt.pcolormesh(covs[v])
    # plt.colorbar()
    # plt.show()
    # ii += 1
    



