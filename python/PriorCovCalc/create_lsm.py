#!/usr/bin/env python3

import xarray as xr
import numpy as np
from global_land_mask import globe
import os


def create_lsm(latmin, latmax, lonmin, lonmax):
     """Creates a land-sea mask (land=1, sea=0) in 0.1 deg x 0.1 deg resolution
     for the chosen area.
     """
     lat = np.linspace(latmin,latmax, (latmax-latmin)*10+1)#, dtype=np.float32)
     lon = np.linspace(lonmin,lonmax, (lonmax - lonmin)*10+1)#, dtype=np.float32)
     longrid, latgrid = np.meshgrid(lon,lat)
     lsm = globe.is_land(latgrid, longrid)
   
     return lsm, latgrid, longrid

def pick_coords(lsm, latgrid, longrid, v):
    if v == "land":
        latv = latgrid[lsm].flatten()
        lonv = longrid[lsm].flatten()
        
                         
    elif v == "ocean":
        latv = latgrid[~lsm].flatten()
        lonv = longrid[~lsm].flatten()
    
    return latv, lonv    


def write_coords_to_file(latv, lonv, v):
    coords = np.column_stack(lonv, latv)

    
#output 
OUTPUT_PATH = '/home/pietaril/Documents/data/coords'
#OUTPUT_PATH = '/scratch/project_462000459/maija/data/co2_prior_unc_cov'


# latmin = 30.0
# latmax = 40.0
# lonmin = -10.0
# lonmax = 0.0

#Europe
latmin = 30
latmax = 75
lonmin = -15
lonmax = 40


lsm, latgrid, longrid = create_lsm(latmin, latmax, lonmin, lonmax)

for v in ["land", "ocean"]:
    latv, lonv = pick_coords(lsm, latgrid, longrid, v)
    print(len(latv))
    print(len(lonv))
    out = xr.Dataset(
        data_vars={"lon": (lonv), "lat": (latv),},
        attrs={'comment': f"Coordinates for {v} cells on a 0.1 x 0.1 deg grid over Europe"})
    fname = f"{v}_coords.nc"
    out.to_netcdf(os.path.join(OUTPUT_PATH, fname))
    

