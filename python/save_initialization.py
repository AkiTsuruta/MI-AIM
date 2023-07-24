#!/usr/bin/env python3

"""Program to save the test data created by functions initialize_state and
initialize_obs from kf.py (in the case when filename = None) so that the same data
can be then used to run kf with different matrix inversion functions and useKG values. """

import os
import numpy as np
import xarray as xr
from kf_playground import initialize_state, initialize_obs

def write_simdata_to_file(filename,xb,B,t,y,R, comments):
    out = xr.Dataset()
    out['state'] = (('nstate'),xb)
    out['cov'] = (('nstate','nstate'), B)
    out['time'] = ('time',t)
    out['obs'] = ('time',y)
    out['R'] = (('time','time'),R)
    out.attrs["comments"] = comments
    out.to_netcdf(filename)


if __name__ == "__main__":

    nstate = None   # number of states
    nobs = 500  # number of observations for the whole time
    tw = 100  # length of time window
    
    x_mu = 1    # state mean values
    y_mu = 1800 # observation mean values
    x_std = None # state uncertainty
    y_std = 15  # obs. uncertainty

    bio_or_anth = "anth"
    file = f'regions_verify_202104_cov.nc'
    fileB = f"data/{file}"

    comments = f'prior cov: {file}, {bio_or_anth}, nstate: {nstate}, nobs: {nobs}, tw: {tw}, x_mu: {x_mu}, y_mu: {y_mu}, x_std: {x_std}, y_std: {y_std}'

    
    # Simulate data
    xb, B, nstate = initialize_state(nstate, x_mu, x_std, fileB=fileB, bio_or_anth=bio_or_anth)
    t, y, R, nobs = initialize_obs(nobs,y_mu,y_std)

    # Write simulated data to netCDF file
    i = 3
    newdir = f'simulated_data/with_real_prior_cov/simulation_{i:02d}'
    os.mkdir(newdir)
    wfile = f'{newdir}/s{i:02d}_init.nc'
    write_simdata_to_file(wfile, xb, B, t, y, R, comments)