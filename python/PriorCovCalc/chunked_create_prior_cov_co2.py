#!/usr/bin/env python3
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import os
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from global_land_mask import globe
import dask.array as da




def haversine_blockwise(lat1, lon1, lat2, lon2):
    """
    Calculate a block of a Haversine pairwise distance matrix.
    For blocks on the diagonal lat1=lat2, lon1 = lon2.

    Parameters:
    - lat1 (ndarray) : 1-d flattened latitude grid
    - lon1 (ndarray) : 1-d flattened longitude grid
    - lat2 (ndarray) : 1-d flattened latitude grid
    - lon2 (ndarray) : 1-d flattened longitude grid
    Returns:
    - distances (ndarray): Pairwise distances matrix.
    """
    # Earth radius in kilometers
    earth_radius = 6371.0
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat = lat1[:, np.newaxis] - lat2
    dlon = lon1[:, np.newaxis] - lon2
    a = da.sin(dlat / 2.0) ** 2 + da.cos(lat2) * da.cos(lat1[:, np.newaxis]) * da.sin(dlon / 2.0) ** 2
    c = 2 * da.arctan2(da.sqrt(a), da.sqrt(1 - a))

    # Calculate pairwise distances in kilometers
    distances = earth_radius * c

    return distances



def create_lsm(latmin, latmax, lonmin, lonmax):
     """Creates a land-sea mask (land=1, sea=0) in 0.1 deg x 0.1 deg resolution
     for the chosen area.
     """
     lat = np.linspace(latmin,latmax, (latmax-latmin)*10+1)#, dtype=np.float32)
     lon = np.linspace(lonmin,lonmax, (lonmax - lonmin)*10+1)#, dtype=np.float32)
     longrid, latgrid = np.meshgrid(lon,lat)
     lsm = globe.is_land(latgrid, longrid)
   
     return lsm, latgrid, longrid

def pick_coords(lsm, latgrid, longrid, v, nchunks = 2):
    if v == "land":
        latv = latgrid[lsm].flatten()
        lonv = longrid[lsm].flatten()
        
                         
    elif v == "ocean":
        latv = latgrid[~lsm].flatten()
        lonv = longrid[~lsm].flatten()
    
    n = len(latv)

    latv = da.from_array(latv, chunks = int((n+1)/nchunks))
    lonv = da.from_array(lonv, chunks= int((n+1)/nchunks))
    
    return latv, lonv    

def compute_cov_blockwise(dists, L, sigmas, v):
    sigma = sigmas[v]
    l = L[v]    
    covv = sigma**2*da.exp(-1*dists/l)
    return covv


#Europe
# latmin = 30
# latmax = 75
# lonmin = -15
# lonmax = 40

#smaller testarea
latmin = 30
latmax = 50
lonmin = -10
lonmax = 0

# Uncertainty (std)
sigmas = {'land': 0.8,
          'ocean': 1.2} 


# Correlation length (km)
L = {'land': 100,  # 
     'ocean': 500}  # for ocean

v = "land"

today = datetime.now()
output_filename = f'test_{v}_{today.strftime("%d%m%Y")}.nc'
#OUTPUT_PATH = '/home/pietaril/Documents/data/unc_cov_matrices'
OUTPUT_PATH = '/scratch/project_462000459/maija/data/co2_prior_unc_cov'


lsm, latgrid, longrid = create_lsm(latmin, latmax, lonmin, lonmax)
latv, lonv = pick_coords(lsm, latgrid, longrid, v, nchunks=2)
nb = latv.blocks[0].shape[0]
covv = da.empty((len(latv), len(latv)), chunks = (nb, nb))
dists =  da.empty((len(latv), len(latv)), chunks = (nb, nb))
#trying out the blockwise function but for the whole matrix
dists[:,:] = haversine_blockwise(latv, lonv, latv, lonv)
covv[:,:] = compute_cov_blockwise(dists, L, sigmas, v)
ds = xr.DataArray(covv)
print(ds)
#w = ds.to_netcdf(os.path.join(OUTPUT_PATH, output_filename), compute=False)
#w.compute()