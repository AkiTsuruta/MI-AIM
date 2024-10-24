
import os
import numpy as np
import pandas as pd
import sys
from pyproj import CRS, Transformer
from skgstat import Variogram
import xarray as xr

#sys.path.append('../src/')

import auxiliar_functions as aux

def create_grid(aoi, res=0.1):

    grid_xy = np.ogrid[aoi[0]:aoi[1]+res:res, aoi[2]:aoi[3]+res:res]
    grid_x, grid_y = grid_xy[0].reshape(1,-1)[0], grid_xy[1][0]
    coords = np.array([(i,j) for i in grid_x for j in grid_y])
    lon_rg, lat_rg = coords[:,0], coords[:,1]
    
    return lon_rg, lat_rg



def read_data(month, UNC_PATH, XCO2_PATH):
    uncertainties_1month = np.load(os.path.join(UNC_PATH, f'rg_uncert_{month}.npy'))
    
    #add cloud cover from xco2 to the months where it is missing
    if month in ["Mar", "Jun", "Jul"]:
        xco2_1month = np.load(os.path.join(XCO2_PATH, f'rg_xco2_{month}.npy'))
        uncertainties_1month[np.isnan(xco2_1month)] = np.nan
  
    return uncertainties_1month

def create_dataset(covariance, V, lon, lat, domain, month):
    
    """
    Creates an xarray dataset of the covariance matrix and coordinates of the grid.
    """

    out_cov = xr.Dataset(
    data_vars={"covariance": (["dim0", "dim1"], covariance),
               "lon": (lon),
                "lat": (lat),},
    attrs={'domain': domain,
           'month' : month,
           'variogram_params' : str(V.describe()), 
            'comment': "XCO2 observation uncertainty covariance matrix computed by variogram approach. Data: version 2 of synthetic CO2M dataset from COCO2 project",})
    
    return out_cov

def compute_variogram(uncertainties, aoi, res, model, maxlag, n_lags, west, lonsplit):
    """
    input:
    - uncertainties: gridded uncertainties for a given time period
    - aoi: area of interest
    - res: resolution of the grid
    - model: variogram model
    - maxlag: maximum distance between points
    - n_lags: number of lags to consider
    """
    
    print("Computing variogram...")

    # create a grid of coordinates

    lon_rg, lat_rg = create_grid(aoi, res)

    # save data to a pandas dataframe with the date and the coordinates
    data = {'lon': lon_rg, 
            'lat': lat_rg, 
            'uncertainties': uncertainties}
    df = pd.DataFrame(data)

    # remove rows with NaN values
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['lon', 'lat'], inplace=True)

    #convert lon, lat to Universal Transverse Mercator (UTM) coordinates
    # NOTE: the projection can be changed to the one that fits the region of interest
    inProj = CRS("EPSG:4326") # WGS84
    outProj = CRS("EPSG:3035")
    transformer = Transformer.from_crs(inProj, outProj, always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    # Add the x,y coordinates to the dataframe
    df['x'] = x
    df['y'] = y


    #compute the variogram for the area of aoi west or east of lonsplit
    if west:
        split_df =  df[df['lon']< lonsplit].copy()
        split_aoi = [aoi[0],lonsplit, aoi[2], aoi[3]]
    
    else:
        split_df = df[df['lon']>= lonsplit].copy()
        split_aoi = [lonsplit, aoi[1], aoi[2], aoi[3]]
    

    split_df.reset_index(drop=True, inplace=True)   
    

    # calculate the empirical variogram
    coordinates = split_df[['x', 'y']].values  # UTM coordinates
    values = split_df['uncertainties'].values  # variable of interest

    V = Variogram(coordinates, values, model=model, maxlag=maxlag, n_lags=n_lags)

    print(f"variogram computed for domain {split_aoi}")

    return V, split_df, split_aoi


def covariance_matrix(V, df):
    """
    input:
    - V: variogram
    - df: dataframe with the data
    
    """

    sill = V.describe()['sill']
    max_range = V.describe()['effective_range']
    alpha = V.describe()['shape']

    # calculate the distances between the points
    coordinates = df[['x', 'y']].values
    distances = aux.calculate_distances(coordinates)
    # build the distance matrix
    W = aux.build_weight_matrix(distances, sill, max_range, alpha)

    # compute covariance matrix
    n = len(df)
    
    print("Computing covariance matrix of sixe ", n, "x", n, "...")

    #note: elementwise multiplication, i.e. "*" on purpose
    cov_matrix = W*np.outer(df["uncertainties"].values, df["uncertainties"].values)

    print("covariance matrix computed")

    return cov_matrix

def covariance_matrix_laia(V, df, OUT_PATH='../output/'):
    """
    input:
    - V: variogram
    - df: dataframe with the data
    - OUT_PATH: path to save the covariance matrix
    
    """

    sill = V.describe()['sill']
    max_range = V.describe()['effective_range']
    alpha = V.describe()['shape']

    # calculate the distances between the points
    coordinates = df[['x', 'y']].values
    distances = aux.calculate_distances(coordinates)
    # build the distance matrix
    W = aux.build_weight_matrix(distances, sill, max_range, alpha)

    # compute covariance matrix
    n = len(df)
    cov_matrix = np.zeros((n,n))
    
    print("Computing covariance matrix of sixe ", n, "x", n, "...")

    for i in range(n):
        for j in range(n):
            if i <= j:
                cov_matrix[i, j] = W[i, j]*df.uncertainties[i]*df.uncertainties[j]
                cov_matrix[j, i] = cov_matrix[i, j]

    print("covariance matrix computed")

    return cov_matrix




def main ():
    # change these parameters
    date = 0
    aoi = [-15, 40, 35, 60] 
    monthid = 6 # which month (Jan=0, Dec=11)
    lonsplit= 2 # longitude along which to split the domain in two

    UNC_PATH = "/home/pietaril/Documents/data/CO2M_simulations/regridded_monthly_uncertainties/"
    XCO2_PATH = "/home/pietaril/Documents/data/CO2M_simulations/regridded_monthly_xco2/"
    OUT_PATH = "/home/pietaril/Documents/data/CO2M_simulations/obs_unc_matrices/"

    #UNC_PATH = '/scratch/project_462000459/maija/data/CO2M_simulations/regridded_monthly_uncertainties/'
    #XCO2_PATH = '/scratch/project_462000459/maija/data/CO2M_simulations/regridded_monthly_xco2/' 

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month = month_names[monthid]
    res = 0.1
    model = 'stable'
    maxlag = 700000
    n_lags = 20
    

    uncertainties = read_data(month, UNC_PATH, XCO2_PATH)
    # compute the variogram

    for west in [True, False]:

        V, split_df, split_aoi = compute_variogram(uncertainties, aoi, res, model, maxlag, n_lags, west, lonsplit)

        # compute the covariance matrix
        cov_matrix = covariance_matrix(V, split_df)

        domain = f"lon {split_aoi[0]}:{split_aoi[1]}, lat {split_aoi[2]}:{split_aoi[3]}"

        lon = split_df["lon"]
        lat = split_df["lat"]

        output_filename = f"obs_unc_cov_{month}_{domain}_{model}_maxlag_{maxlag}_nlags_{n_lags}.nc"
        ds_out = create_dataset(cov_matrix, V, lon, lat, domain, month)
        ds_out.to_netcdf(os.path.join(OUT_PATH, output_filename))    


if __name__ == "__main__":
    main()