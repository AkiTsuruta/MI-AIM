# version with Maija's comments 


from datetime import date
from nearestPD import nearestPD
from functions import points2distance, find_centre_region
import xarray as xr
import numpy as np
import sys
# sys.path.append('../regions/')
# sys.path.append('../pyfunc/')

today = date.today()

region = xr.open_dataset('/home/pietaril/Documents/MI-AIM/python/PriorCovCalc/regions_verify_202011.nc')
transcom_regions = region.transcom_regions.values
ntc = transcom_regions.max()

#northern high latitudes zoom-in 1 x 1 degree
nhl = list(range(1, 7)) + [15, 16] + list(range(23, 30))
print('NHL 1x1 mTCs: ', nhl)

# find centre of each TC region
latc_tc, lonc_tc = find_centre_region(
    region.latitude.values, region.longitude.values,
    transcom_regions, transcom_regions,
    ntc)

# Uncertainty, diagnals
sigmas = {'land': 0.8,
          'ocean': 0.2,
          'ice': 1e-8}
oce_tc = list(range(30, ntc))
ice_tc = ntc

# Correlation length (km)
L = {'land1': 100,  # for 1x1
     'land2': 500,  # for mTC
     'ocean': 900}  # for ocean

# Output dataset
out_cov = xr.Dataset()
wfile = 'regions_verify_%04d%02d.nc' % (today.year, today.month)


def sigma(tc, ntc, var, sigmas, oce_tc, ice_tc):
    out = sigmas['land'] if tc not in oce_tc else sigmas['ocean']
    if tc == ice_tc:
        out = sigmas['ice']
    return out


def latlon11(region, r, v):
    w = np.where(region['regions_%s' % v].values == r)
    latc = region.latitude.values[w[0]]
    lonc = region.longitude.values[w[1]]
    return latc, lonc


def gettc(region, j=None):
    if j == None:
        tc = np.array(list(set(region.transcom_regions.values.flatten())))
        tc = tc[~np.isnan(tc)]
    else:
        w = np.where(region['regions_%s' % v].values == j)
        tc = list(set(region.transcom_regions.values[w].flatten()))
        if len(tc) != 1:
            print('[ERROR] len(tc) not equal to 1!')
            sys.exit()
    tc = int(tc[0])
    return tc


def calc_offdiagnals11(cov, region, j, v, dregions, L):
    # off-diagonals for 1x1 areas
    latc1, lonc1 = latlon11(region, j, v)
    # off-diagnals
    for jj in dregions:
        # tämä varmaan siksi että ei kahdesti lasketa samaa off-diagonaalia
        if j >= jj:
            continue
        jj = int(jj)
        latc2, lonc2 = latlon11(region, jj, v)
        dists = points2distance(latc1, lonc1, latc2, lonc2)
        # print(j,jj, latc1,lonc1,latc2,lonc2,dists);sys.exit()
        cov[j-1, jj-1] = cov[j-1, j-1]*np.exp(-1*(dists/L['land1']))
    return cov


def istccorr(tc, tc2, nhl, oce_tc):
    # Whether tc and tc2 are correlated
    out = True
    if tc >= tc2:
        out = False
    if tc2 in nhl:
        out = False
    if tc not in oce_tc and tc2 in oce_tc:
        out = False  # no correlation between land and ocean
    if tc in oce_tc and tc2 not in oce_tc:
        out = False  # no correlation between land and ocean
    return out

def calc_offdiagnalstc(cov, tc, ntc, nhl, oce_tc, L, latc_tc, lonc_tc, region,
                       transcom_regions, v):
    #Version that Maria has fixed. Off-diagonals of TransCom areas
    for tc2 in range(1, ntc):  # do not loop through ice region
        if istccorr(tc, tc2, nhl, oce_tc):
            # correlation length
            Ld = L['land2'] if tc not in oce_tc else L['ocean']
            dists = points2distance(
                latc_tc[tc-1], lonc_tc[tc-1], latc_tc[tc2-1], lonc_tc[tc2-1])
            jj = list(set(region['regions_%s' % v].values[np.where(
                transcom_regions == tc2)].flatten()))[0]
            # print('        ',tc,tc2,j-1,ind)
            cov[j-1, jj-1] = ((cov[j-1, j-1]**0.5) * (cov[jj-1, jj-1]**0.5)
                              * np.exp(-1*(dists/Ld)))
    return cov

# alla olevassa calc_offdiagnalstc Marian mukaan bugi

# def calc_offdiagnalstc(cov, tc, ntc, nhl, oce_tc, L, latc_tc, lonc_tc, region,
#                        transcom_regions, v):
#     for tc2 in range(1, ntc):  # do not loop through ice region
#         if istccorr(tc, tc2, nhl, oce_tc):
#             # correlation length
#             Ld = L['land2'] if tc not in oce_tc else L['ocean']
#             dists = points2distance(
#                 latc_tc[tc-1], lonc_tc[tc-1], latc_tc[tc2-1], lonc_tc[tc2-1])
#             jj = list(set(region['regions_%s' % v].values[np.where(
#                 transcom_regions == tc2)].flatten()))[0]
#             # print('        ',tc,tc2,j-1,ind)
#             cov[j-1, jj-1] = cov[j-1, j-1]*np.exp(-1*(dists/Ld))
#     return cov


def finalize(cov, sigmas):
    # If matrix is not positive definite, take nearest PD
    try:
        # C = np.linalg.cholesky(cov)
        np.linalg.cholesky(cov)

    except:
        print('Maxrix not positive definite. Take nearest PD.')
        # The matrix is probably not postive definite
        cov = nearestPD(cov)
        cov[-1, -1] = sigmas['ice']  # ice, make sure it's same as defined
    return cov


def check(cov):
    dof = 0
    # test
    try:
        C = np.linalg.cholesky(cov)
        _, s, _ = np.linalg.svd(cov)
        dof = np.sum(s) ** 2 / sum(s ** 2)
        print('Approx. dof:', int(dof))
    except:
        print('[ERROR] Matrix not positive definite!')
        sys.exit()
    return dof


def write_dataarray(data, cov, v, nr, dof):
    data['covariance_%s' % v] = xr.DataArray(
        data=cov,
        dims=['nparams_%s' % v, 'nparams_%s' % v],
        coords={'nparams_%s' % v: np.arange(
            nr), 'nparams_%s' % v: np.arange(nr)},
        attrs={
            'comment': "Prior covariance for %s fluxes. Approx. dof = %s" % (v, int(dof))}
    )
    return data


ncount = range(0, 7000, 100)
for vv in ['bio', 'anth', 'anth2']:
    v = 'bio' if vv == 'bio' else 'anth'
    print('Create covariance for ', v)

    nr = region['regions_%s' % v].values.max()
    cov = np.zeros((nr, nr))
    categ = region['regions_%s_categ' % v]

    for i in set(categ.values.flatten()):
        print('  categ: ', i)
        # filter variables in dataset "region" so that everything for which categ not i is set to nan
        dummy = region.where(categ == i, drop=True)
        # get tc corresponding to categ i
        tc = gettc(dummy, None)

        if v == 'bio' and tc in oce_tc:
            print(tc)
            continue  # do not optimize ocean flux for bio
        
        #filter optimization regions corresponding to categ i
        dregions = dummy['regions_%s' % v].values.flatten()
        dregions = set(dregions[~np.isnan(dregions)])

        for j in dregions:
            j = int(j)
            if j in ncount:
                print(j)

            # diagnals
            tc = gettc(dummy, j)
            # j-1 because opt regions numbering starts from 1 -> conversion to 0-based indexing
            cov[j-1, j-1] = sigma(tc, ntc, v, sigmas, oce_tc,
                                  ice_tc)**2
            if tc == ice_tc:
                print('        Ice. Independent of any other regions')
                continue

            if vv == 'anth2':
                continue  # no correlation at all
            # off-diagnals
            if i == 0:  # 1x1
                cov = calc_offdiagnals11(cov, dummy, j, v, dregions, L)
            else:  # mTCs
                if v == 'anth':
                    continue  # no correlation over mTCs in anth
                cov = calc_offdiagnalstc(
                    cov, tc, ntc, nhl, oce_tc, L, latc_tc, lonc_tc, region, transcom_regions, v)
            break
        break
    break

# i_lower = np.tril_indices(nr, -1) # lower triangular indices (excluding the diagonal)
# cov[i_lower] = cov.T[i_lower]

# cov = finalize(cov, sigmas)
# dof = check(cov)
# out_cov = write_dataarray(out_cov, cov, vv, nr, dof)

# out_cov.to_netcdf('%s_cov.nc' % wfile.split('.nc')[0])
