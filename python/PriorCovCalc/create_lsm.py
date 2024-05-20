#!/usr/bin/env python3

import xarray as xr
import numpy as np
from global_land_mask import globe
import os

def create_lsm(latmin, latmax, lonmin, lonmax):
     """Creates a land-sea mask (land=1, sea=0) in 0.1 deg x 0.1 deg resolution
     for the chosen area.
     """
     lat = np.linspace(latmin,latmax, (latmax-latmin)*10+1, dtype=np.float32)
     lon = np.linspace(lonmin,lonmax, (lonmax - lonmin)*10+1, dtype=np.float32)
     print("creating longrid, latgrid")
     longrid, latgrid = np.meshgrid(lon,lat)
     print("computing lsm")
     lsm = globe.is_land(latgrid, longrid)
   
     return lsm, latgrid, longrid


#output 
#OUTPUT_PATH = '/home/pietaril/Documents/data/unc_cov_matrices'
OUTPUT_PATH = '/scratch/project_462000459/maija/data/co2_prior_unc_cov'


# latmin = 30.0
# latmax = 40.0
# lonmin = -10.0
# lonmax = 0.0

latmin = 30.0
latmax = 75.0
lonmin = -15.0
lonmax = 40.0

fname = f"lsm_lat{latmin}_{latmax}_lon{lonmin}_{lonmax}"


lsm, latgrid, longrid = create_lsm(latmin, latmax, lonmin, lonmax)

