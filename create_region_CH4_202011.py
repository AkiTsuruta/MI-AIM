#
# Create cov. and regions file
#
# Assume NHL and Europe as 1x1, elsewhere region
#

import xarray as xr 
import numpy as np
import matplotlib.pylab as plt
from datetime import date 

today = date.today()

#---------------------
# Create regional definition
#---------------------
orig = xr.open_dataset('../regions/regions_verify_202011.nc')
mtc = orig.transcom_regions.values
orig['latitude'] = np.arange(-90,90)+0.5
orig['longitude'] = np.arange(-180,180)+0.5

nhl = list(range(1,7)) + [15,16] + list(range(23,30))
print('NHL 1x1 mTCs: ', nhl)

out = {}
for v in ['bio','anth']:
    for vv in ['','_categ']:
        out['regions_%s%s'%(v,vv)] = np.zeros((180,360))

for v in ['bio','anth']:
    i = 1
    for ind_mtc in set(mtc.flatten()):
        w = np.where(mtc==ind_mtc)
        if ind_mtc not in nhl:
            if ind_mtc==30: categ_ocnind = i
            out['regions_%s_categ'%v][w] = i
            i = i+1
print('Ocean index (category)', categ_ocnind)

for v in ['bio','anth']:
    i = 1
    categ = out['regions_%s_categ'%v]
    for ind_categ in set(categ.flatten()):
        w = np.where(categ==ind_categ)
        if ind_categ == 0:
            for k in range(len(w[0])):
                out['regions_%s'%v][w[0][k],w[1][k]] = i
                i = i+1
        else:
            if ind_categ == categ_ocnind: reg_ocnind = i
            out['regions_%s'%v][w] = i
            i = i+1
print('Ocean index (region)', reg_ocnind)

# Check by plotting
# for i,v in enumerate(out):
#     print(out[v].min(), out[v].max())
#     plt.figure(i+1)
#     plt.title(v)
#     plt.pcolor(out[v])
#     plt.colorbar()
#     plt.show()

out_regions = xr.Dataset(attrs={'description':"Covariance related parameters for CTDAS-CH4"})
out_regions['transcom_regions'] = orig.transcom_regions
for v in ['bio','anth']:
    for vv in ['','_categ']:
        if vv == '':
            attrtext = 'Optimization regions for %s fluxes.'%v 
        else:
            attrtext = 'Categories of optimization regions. For high northern mTCS, optimization is 1x1 (categ=0), elsewhere global is per mTCs. Note the numbers do not correspond to mTC.'
        out_regions['regions_%s%s'%(v,vv)] = xr.DataArray(
            data = out['regions_%s%s'%(v,vv)].astype(int),
            dims = {'latitude':orig.latitude, 'longitude':orig.longitude},
            attrs = {'comment':attrtext}
        )
wfile = 'regions_verify_%04d%02d.nc'%(today.year,today.month)
out_regions.to_netcdf(wfile)

#---------------------
# Covariance
#---------------------
import math
import sys
sys.path.append('../regions/')
#from functions import points2distance, find_centre_region
sys.path.append('../pyfunc/')
from nearestPD import nearestPD

def points2distance(lat1,lon1,lat2,lon2):
  """
    Calculate distance (in kilometers) between two points given as (lon, lat) pairs
    based on Haversine formula.
  """
  start_long = math.radians(lon1)
  start_latt = math.radians(lat1)
  end_long = math.radians(lon2)
  end_latt = math.radians(lat2)
  d_latt = end_latt - start_latt
  d_long = end_long - start_long
  a = math.sin(d_latt/2)**2 + math.cos(start_latt) * math.cos(end_latt) * math.sin(d_long/2)**2
  c = 2 * math.atan2(math.sqrt(a),  math.sqrt(1-a))
  return 6371 * c


def find_centre_region(lat,lon,regions,tcregions,Tice):
   # Find centre of all regions

   set_reg = set(regions.flatten())
   #rmax = int(regions.max())
   latc = []; lonc = []

   for i in set_reg: #range(0,rmax+1):
      w = np.where(regions==i)

      if tcregions[w][0] == Tice:  #Ice
         print('Calculate mid point differently: TC=', tcregions[w][0])
         lonc.append(0.)
         latc.append(-85.5)

      else: # Other than Ice

         latc.append((lat[w[0]].max()+lat[w[0]].min())/2.)
      
         dummy = lon[w[1]]
         if (-179.5 in dummy)&(179.5 in dummy):
            print('Calculate mid point differently: TC=', tcregions[w][0])
            kw = np.where(dummy<0)[0]
            dummy[kw] = dummy[kw]+360
            k = (dummy.max()+dummy.min())/2.
            if k > 179.5:
              k = k-360
            lonc.append( k )
         else:
            lonc.append((lon[w[1]].max()+lon[w[1]].min())/2.)
      
   return latc,lonc



lat = orig.latitude.values
lon = orig.longitude.values

sigma_land = 0.8
sigma_ocn = 0.2
sigma_ice = 1e-8

L_land = 500
L_ocn = 900

cov = {}
for v in ['bio','anth']:
    regions = out['regions_%s'%v]
    categ =  out['regions_%s_categ'%v]

    nregions = len(list(set(regions.flatten())))
    cov[v] = np.zeros((nregions,nregions))

    #---- diagnals
    for ind_reg in range(nregions):
        w = np.where(regions==ind_reg+regions.min()) #regions.min = 1
        cov[v][ind_reg,ind_reg] = sigma_ocn if list(set(mtc[w]))[0] >= 30 else sigma_land
        if list(set(categ[w]))[0] == categ.max(): cov[v][ind_reg,ind_reg] = sigma_ice #ice

    #---- off diagnals
    # find centre of each region
    latc,lonc = find_centre_region(lat,lon,regions,mtc,mtc.max())
    
    for i in range(nregions-1): #no correlation for ice
        for j in range(nregions-1): #no correlation for ice
            if i == j: continue
            # calculate distance in km
            dists = points2distance(latc[i],lonc[i],latc[j],lonc[j])
            if i<reg_ocnind and j<reg_ocnind: #land
                cov[v][i,j] = sigma_land*math.exp(-1*(dists/L_land))
            if i>reg_ocnind and j>reg_ocnind: #ocean
                cov[v][i,j] = sigma_ocn*math.exp(-1*(dists/L_ocn))

    # If matrix is not positive definite
    try: 
        C = np.linalg.cholesky(cov[v])
    except:
        # The matrix is probably not postive definite
        cov[v] = nearestPD(cov[v])
        cov[v][-1,-1] = sigma_ice #ice, make sure it's same as defined

    # test
    try: 
        C = np.linalg.cholesky(cov[v])
    except:
        print('[ERROR] Matrix not positive definite!')
        sys.exit()

    _, s, _ = np.linalg.svd(a.covariance_bio.values)  
    dof =  np.sum(s) ** 2 / sum(s ** 2) 
    print('Approx. dof:', int(dof))


# Write to file
out_cov = xr.Dataset()
for v in ['bio','anth']:
    s = cov[v].shape[0]
    #out_cov.expand_dims({'nparams_%s'%v:s})
    out_cov['covariance_%s'%v] = xr.DataArray(
        data = cov[v],
        dims = ['nparams_%s'%v,'nparams_%s'%v],
        coords = {'nparams_%s'%v:np.arange(s),'nparams_%s'%v:np.arange(s)},
        attrs= {'comment':"Prior covariance for %s fluxes."%v}
    )
out_cov.to_netcdf('%s_cov.nc'%wfile.split('.nc')[0])
