import math
import sys
import numpy as np
import copy


def join_tc_eco(tc, eco, pbio, start_ocean, ice):

    s = tc.shape
    region = np.zeros((s[0], s[1]))
    bioland = np.zeros((s[0], s[1]))
    ffland = np.zeros((s[0], s[1]))
    ocereg = np.zeros((s[0], s[1]))
    nbio = []

    w = np.where((eco >= pbio[0]) & (eco <= pbio[1]))
    bioland[w] = 1

    w = np.where(bioland == 0)
    ffland[w] = 1

    w = np.where((tc >= start_ocean) & (tc != ice))
    ocereg[w] = tc[w]-start_ocean+1

    ss = 0
    for j in range(int(tc.max())):
        w = np.where(tc == j+1)
        ss = ss + len(list(set(list(eco[w]))))

    k = 0
    # bio regions
    for i in range(pbio[0], pbio[1]+1):
        for j in range(int(tc.max())):
            w = np.where((tc == j+1) & (eco == i))
            if len(w[0]) != 0:
                k = k+1
                region[w] = k
        nbio.append(k)

    # anth
    for i in range(pbio[1]+1, int(eco.max())+1):
        for j in range(int(tc.max())):
            w = np.where((tc == j+1) & (eco == i))
            if len(w[0]) != 0:
                k = k+1
                region[w] = k

    # ocean
    for j in range(int(tc.max())):
        w = np.where((tc == j+1) & (eco == 0))
        if len(w[0]) != 0:
            k = k+1
            region[w] = k

    if ss != k:
        print('number of regions do not much list', ss, k)
        sys.exit()

    return region, bioland, ffland, ocereg, nbio


def points2distance(lat1, lon1, lat2, lon2):
    """
      Calculate distance (in kilometers) between two points given as
      (lon, lat) pairs based on Haversine formula.
    """
    start_long = math.radians(lon1)
    start_latt = math.radians(lat1)
    end_long = math.radians(lon2)
    end_latt = math.radians(lat2)
    d_latt = end_latt - start_latt
    d_long = end_long - start_long
    a = math.sin(d_latt/2)**2 + math.cos(start_latt) * \
        math.cos(end_latt) * math.sin(d_long/2)**2
    c = 2 * math.atan2(math.sqrt(a),  math.sqrt(1-a))
    return 6371 * c


def find_centre_region(lat, lon, regions, tcregions, Tice):
    # Find centre of all regions

    set_reg = set(regions.flatten())
    #  rmax = int(regions.max())
    latc = []
    lonc = []

    for i in set_reg:  # range(0,rmax+1):
        w = np.where(regions == i)

        if tcregions[w][0] == Tice:  # Ice
            print('Calculate mid point differently: TC=', tcregions[w][0])
            lonc.append(0.)
            latc.append(-85.5)

        else:  # Other than Ice

            latc.append((lat[w[0]].max()+lat[w[0]].min())/2.)

            dummy = lon[w[1]]
            if (-179.5 in dummy) & (179.5 in dummy):
                print('Calculate mid point differently: TC=', tcregions[w][0])
                kw = np.where(dummy < 0)[0]
                dummy[kw] = dummy[kw]+360
                k = (dummy.max()+dummy.min())/2.
                if k > 179.5:
                    k = k-360
                lonc.append(k)
            else:
                lonc.append((lon[w[1]].max()+lon[w[1]].min())/2.)

    return latc, lonc


def find_centre_region2(lat, lon, regions, i, tcregions, Tice):
    # Find centre of a specific region

    w = np.where(regions == i)

    # a region
    if len(w[0]) > 1:
        if tcregions[w][0] == Tice:  # Ice
            print('Calculate mid point differently: TC=', tcregions[w][0])
            lonc = 0.
            latc = -85.5

        else:  # Other than Ice
            latc = (lat[w[0]].max()+lat[w[0]].min())/2.

            dummy = lon[w[1]]
            if (-179.5 in dummy) & (179.5 in dummy):
                print('Calculate mid point differently: TC=', tcregions[w][0])
                kw = np.where(dummy < 0)[0]
                dummy[kw] = dummy[kw]+360
                k = (dummy.max()+dummy.min())/2.
                if k > 179.5:
                    k = k-360
                lonc = k
            else:
                lonc = (lon[w[1]].max()+lon[w[1]].min())/2.

    # one grid
    else:
        latc = lat[w[0]]
        lonc = lon[w[1]]

    return latc, lonc
