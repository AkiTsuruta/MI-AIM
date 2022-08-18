import numpy as np
import xarray as xr


def dechunk_var(da, variable:str, tw:int, ntw = 5):
    '''Helper function to "de-chunk" a variable from kf output
    by combining non-na values from time window chunks together 
    into a single timeline to make plotting simpler.
    '''
    var_in_chunks = da[variable]
    ls = []
    for i in range(ntw):
        ls.append(var_in_chunks[i,i*tw:(i+1)*tw])
    flat_var = xr.concat(ls, dim = 'time')
    return flat_var

def diff_post(da):
    '''Function to calculate obs-post difference from
    given data in a similar way as obs-prior differences
    are calculated in kf.py
    '''
    t = da.time.values 
    ntw = da.dims['time2'] #number of timesteps
    tw = int(len(t)/ntw) # length of time window
    xa = da.posterior.values
    y = dechunk_var(da, 'obs', tw)
    nobs = len(y)
    nstate = len(da.nstate)
    
    H = np.ones((nobs,nstate))*1800/nstate

    ls = []
    for timestep in range( int(len(t)/tw) ): #Loop through time
    
        # Select observational data for this time 
        w = np.where((t>=timestep*tw)&(t<timestep*tw+tw))[0]
        y_t = y[w]
        xa_t = xa[timestep,:]
        H_t= H[w,:]
        
        diff_t = y_t - H_t.dot(xa_t)
        ls.append(diff_t)
    
    diff_post = xr.concat(ls, dim = 'time')
    return diff_post