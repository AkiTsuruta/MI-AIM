#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
MAIJAN MUUNTELEMA KOPIO

Basic Kalman filter code to minimize the cost function

J = (x - xb)^T B-1 (x - xb) + (y - H(x))^T R-1 (y - H(x))

x = state to be optimized
xb = prior state
B = prior state error covariance matrix (P-1 = inverse of P)
y = observations
H = observation operator
R = observation error covariance matrix (R-1 = inverse of R)

If H can be linealized and can be written in a matrix form, then H(x) = Hx.

-------------------
1) Use Kalman gain
-------------------
xa = xb + KG(y - Hxb) # Posterior state
A = B - KHB           # Posterior state error covariance matrix
KG = BHT(R + HBHT)-1  # Kalman gain
HT = H^T              # Transpose of H

-------------------
2) Without Kalman gain
-------------------
xa = xb + (HTR-1H + B-1)-1 HTR-1 (y - Hxb) # Posterior state
A = (HTR-1H + B-1)-1                       # Posterior state error covariance matrix

--------------------
Prediction 
--------------------
xb(t+1) = Mxa(t)  # state
B (t+1) = MAMT + Q # covariance
M = forecast model
MT = M^T (transpose of M)
Q = error covariance of the forecast model


Author: @aki
Project: 
Rivision history
File Created: 2022-06-23
"""
import os
import numpy as np
import xarray as xr
from copy import deepcopy
from blockinv_iterative import block_inv

invm = np.linalg.inv # Function to invert matrix

def initialize_state(nstate, x_mu, x_std, filename=None):
    if filename is None: 
        B = np.diag((np.ones(nstate)*x_std))
        xb = np.random.normal(x_mu,x_std,size=nstate)
    else: 
        # Read data from file
        data = xr.open_dataset(filename)
        xb = data.state.values
        B = data.cov.values
        nstate = data.dims['nstate']
    return xb, B, nstate

def initialize_obs(nobs,y_mu,y_std,filename=None):
    if filename is None:
        t = np.arange(nobs) 
        y = np.sin(t) + np.random.normal(loc=y_mu, scale=y_std, size=nobs)   
        R = np.diag((np.ones(nobs)*y_std)) 
    else:
        # Read data from file
        data = xr.open_dataset(filename)
        y = data.obs.values
        t = data.time.values
        R = data.R.values
        nobs = data.dims['time'] # changed from 'nobs' to 'time' to match simulated data
    return t, y, R, nobs

    
def optimize(useKG, xb, B, y, H, R):
    diff = y - H.dot(xb)
    if useKG:
        # Calculate Kalman gain
        S  = R + H.dot(B).dot(H.T)
        Si = invm(S)
        KG = B.dot(H.T).dot(Si)

        # Posteriors
        xa = xb + KG.dot(diff)
        A = B - KG.dot(H).dot(B)
    else:
        # Calcualte some matrices
        Ri = invm(R)
        Bi = invm(B)
        S = (H.T).dot(Ri).dot(H) + Bi

        # Posteriors
        A = invm(S)        
        xa = xb + A.dot(H.T).dot(Ri).dot(diff)

    return xa, A, diff


def predict(xa,A,M=None,Q=None):
    if M is None:
        xb = deepcopy(xa)
        B  = deepcopy(A)
    else:
        xb = M.dot(xa)
        B  = M.dot(A).dot(M.T) + Q
    return xb, B


def write_to_file(filename,xb,xa,B,A,time,y,R,diff):
    out = xr.Dataset()
    out['prior'] = (('nstate'),xb)
    out['posterior'] = (('nstate'),xa)
    out['prior_cov'] = (('nstate','nstate'), B)
    out['posterior_cov'] = (('nstate','nstate'), A)
    out['time'] = ('time',time)
    out['obs'] = ('time',y)
    out['obs_unc'] = (('time','time'),R)
    out['differences'] = ('time',diff)
    out.differences.attrs['comments'] = 'Obs - prior differences'
    out.attrs['comments'] = f'Fixed version of H. Matrix inverse function: {invm.__name__} from module: {invm.__module__}. useKG: {useKG}'
    out.to_netcdf(filename)

if __name__ == "__main__":

    nstate = None   # number of states
    nobs = None  # number of observations for the whole time
    tw = 10 # length of time window
    
    x_mu = None    # state mean values
    y_mu = None # observation mean values
    x_std = None # state uncertainty
    y_std = None  # obs. uncertainty

    i = 12 # when using simulated data: simulation number

    fname = f'simulated_data/simulation_{i:02d}/s{i:02d}_init.nc' # file to read

    # initialize values
    xb, B, nstate = initialize_state(nstate, x_mu, x_std, filename=fname)
    t, y, R, nobs = initialize_obs(nobs, y_mu, y_std, filename=fname)

    useKG = True # whether to use Kalman gain or not
    H = np.ones((nobs,nstate))*1800/nstate # dummy observation operator

    #create a folder for output data
    if useKG:
        name_end = "with_kf"
    else:
        name_end = "without_kf"
    if invm.__module__ == "numpy.linalg":
        dirname = f'out_default_{name_end}'
    elif invm.__module__ == "blockinv_iterative":
        dirname = f'out_block_{name_end}'
    else: 
        dirname = f'out_{invm.__name__}_{name_end}'
    newdir = f'simulated_data/simulation_{i:02d}/{dirname}' 
    os.mkdir(newdir)

    for timestep in range( int(len(t)/tw) ): #Loop through time
        # Select observational data for this time 
        w = np.where((t>=timestep*tw)&(t<timestep*tw+tw))[0]
        y_t = y[w]
        R_t = R[w,:][:,w] 
        H_t= H[w,:]

        # Optimize
        xa, A, diff = optimize(useKG, xb, B, y_t, H_t, R_t)

        # Write data to netCDF file
        wfile = f'{newdir}/s{i:02d}_out_{timestep:02d}.nc'
        write_to_file(wfile,xb,xa,B,A,t[w],y_t,R_t,diff)

        # The updated state is prior for next state
        xb, B = predict(xa,A)
