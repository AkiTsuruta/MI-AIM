import numpy as np
# # import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import xarray as xr
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
import time



ds_2 = xr.open_dataset("regions_verify_202104_cov.nc")
bio_2 = ds_2["covariance_bio"]
anth_2 = ds_2["covariance_anth"]

 
start = time.time()

bio_2_inv = inv(csc_matrix(bio_2))

end = time.time()
 
print("The time of execution of sc.sparse.linalg.inv():", end-start)




