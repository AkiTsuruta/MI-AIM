import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import bottleneck as bn



# 41 x 41 -matriisi
ds_1 = xr.open_dataset("regions_verify_isotope_202112_cov.nc")
bio_1 = ds_1["covariance_bio"]




# #  6858 x 6858 -matriisi
ds_2 = xr.open_dataset("regions_verify_202104_cov.nc")
bio_2 = ds_2["covariance_bio"]

import time
 

# Inverse matrix with np standard inv function. bio_inv is now a numpy ndarray, not xarray.DataArray
start = time.time()

bio_inv = np.linalg.inv(bio_1) 

end = time.time()
 
print("The time of execution of np.linalg.inv():", end-start)



"""Eigendecomposition: oma funktio käänteismatriisin laskemiseen eigendecompositionin avulla. 
Olisi varmaan hyvä tarkistaa mikä on nopeuden kannalta fiksuin tapa järjestää tuo koodi, koska tämä on tosi hidas.
Tosi hidas isommalla matriisilla senkin jälkeen kun otin transpoosin ominaisvektoreista (n. 45 s. vs. default funktiolla 4 s)"""

def eigen_inv(A):
    evals, evecs = np.linalg.eig(A)  #eig nopeampi kuin eigh ainakin oman testin mukaan 
    A_inv = evecs@np.diag(1/evals)@np.transpose(evecs) # Symmetriselle pitäisi päteä, että om.vektorien muod. matriisi ortogonaalinen, jolloin sen transpoosi on sen käänt.matriisi
    return A_inv


# start = time.time()
# decomp_inv = eigen_inv(bio_2)
# end = time.time()

# print("The time of computing matrix inverse using eigdecomposition:", end-start)



# print(np.allclose(bio_inv, decomp_inv)) 
# 
#  # Tarkistetaan onko samat. Allclose-metodi, koska pyöristykset aiheuttaa pieniä eroja
# print(np.allclose(decomp_1_inv, bio_1_inv))

