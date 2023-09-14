"""
Created on Wed Jul 5 2023

@author: Laia Amorós
"""

import netCDF4 as nc
import numpy as np

import os
import sys
sys.path.append('../src')

def regrid(data, lon, lat, factor1, factor2):
    """
    Regrids data to lower resolution by averaging over a window of size factor1 x factor2.
    """
    data_unmasked = data.filled(np.nan)
    data_shape = data.shape
    regridded_shape = (data_shape[0] // factor1, data_shape[1] // factor2)

    data_reg = np.full(regridded_shape, np.nan)
    lon_reg = np.zeros(regridded_shape)
    lat_reg = np.zeros(regridded_shape)

    for i in range(regridded_shape[0]):
        for j in range(regridded_shape[1]):
            window = data_unmasked[i*factor1:min((i+1)*factor1, data_shape[0]), 
                                                j*factor2:min((j+1)*factor2, data_shape[1])]
            lon_window = lon[i*factor1:min((i+1)*factor1, data_shape[0]), 
                                                j*factor2:min((j+1)*factor2, data_shape[1])]
            lat_window = lat[i*factor1:min((i+1)*factor1, data_shape[0]), 
                                                j*factor2:min((j+1)*factor2, data_shape[1])]

            data_reg[i, j] = np.nanmean(window)
            lon_reg[i, j] = np.nanmean(lon_window)
            lat_reg[i, j] = np.nanmean(lat_window)

    # Mask new arrays where np.isnan is True
    data_reg = np.ma.masked_where(np.isnan(data_reg), data_reg)

    return data_reg, lon_reg, lat_reg

def covariance_matrix(matrix):
    """
    Computes the covariance matrix of a matrix with NaN values.
    """
    matrix_flat = np.asarray(matrix).flatten()
    # Replace NaN values with mean
    matrix_flat[np.isnan(matrix_flat)] = np.nanmedian(matrix_flat)

    deviations = matrix_flat - np.nanmean(matrix_flat)
    outer_product = np.outer(deviations, deviations)
    
    covariance_matrix = outer_product / (len(matrix_flat) - 1)

    return covariance_matrix


def main():
    NUMBER_OF_FILES = 1 # change to the number of covariance matrices you want to produce
    FACTOR1 = 15
    FACTOR2 = 2

    # Path to CO2M simulations data and output
    # in LUMI
 #   DATA_PATH = '/scratch/project_462000289/CO2M_obs/CO2M_simulations/2018/Orbits_Europe/CO2Meast/'
 #   OUTPUT_PATH = '/scratch/project_462000289/covariance_matrices'
    DATA_PATH = '../../data/CO2M_simulations/2018/Orbits_Europe/CO2Meast/'
    OUTPUT_PATH = '../../output'
    
    filenames = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.nc')]
        
    for file in filenames[:NUMBER_OF_FILES]:
        # Read data
        data_nc = nc.Dataset(file, 'r')
        xco2 = data_nc.groups['data']['observation_data']['xco2'][:]
        xco2_quality_flag = data_nc.groups['data']['observation_data']['xco2_quality_flag'][:]
        uncertainties = data_nc.groups['data']['observation_data']['xco2_precision'][:]
        lon = data_nc.groups['data']['geolocation_data_dem']['longitude'][:]
        lat = data_nc.groups['data']['geolocation_data_dem']['latitude'][:]

        # Regrid data to lower resolution. Factor1 and factor2 can be changed to any integer
        uncertainties_reg, lon_reg, lat_reg = regrid(uncertainties, lon, lat, FACTOR1, FACTOR2)

        # Compute covariance matrix
        covariance = covariance_matrix(uncertainties_reg)

        # Save covariance matrix to output folder
        date = os.path.basename(file)[21:29]
        output_filename = f'uncertainty_covariance_matrix{date}.npy'
        np.save(os.path.join(OUTPUT_PATH, output_filename), covariance)


if __name__ == "__main__":
    main()
