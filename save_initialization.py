"""Program to save the test data created by functions initialize_state and
initialize_obs (in the case when filename = None) so that the same data
can be then used to run kf with different matrix inversion functions. """

import numpy as np
import xarray as xr
from kf import initialize_state, initialize_obs


def write_simdata_to_file(filename,xb,B,t,y,R):
    out = xr.Dataset()
    out['state'] = (('nstate'),xb)
    out['cov'] = (('nstate','nstate'), B)
    out['time'] = ('time',t)
    out['obs'] = ('time',y)
    out['R'] = (('time','time'),R)
    out.to_netcdf(filename)



if __name__ == "__main__":

    nstate = 2   # number of states
    nobs = 50   # number of observations for the whole time
    tw = 10    # length of time window
    
    x_mu = 1    # state mean values
    y_mu = 1800 # observation mean values
    x_std = 0.8 # state uncertainty
    y_std = 15  # obs. uncertainty

    # Simulate data
    xb, B, nstate = initialize_state(nstate, x_mu, x_std)
    t, y, R, nobs = initialize_obs(nobs,y_mu,y_std)

    # Write simulated data to netCDF file
    #wfile = f'simulated_data/simulation_02/init_data'
    write_simdata_to_file(wfile, xb, B, t, y, R)