import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import xarray as xr


# THINGS TO CONSIDER: 
# 
# 1) VOIKO KÄYDÄ NIIN, ETTÄ K:N PYÖRISTYKSEN TAKIA JOKU OSA JÄÄ ULKOPUOLELLE?
#
# 2) 2 iteraatiolla ei taida toimia tuo funktio
#
# 3) Should the conversion of XarrayDataArray to numpy array happen
# inside the function or as a separate instance?
#
# 4) What if the matrix to be inverted (or one of it's submatrices that need
# to be inverted is singular?)


 

def block_inv(A,idx):
    """Function divides 2-D numpy array A (i.e. xarrayDataArrays need to be converted
    to numpy before using this function) into four blocks, so that the lower 
    left-hand corner of the first block is at A[idx-1,idx-1]. Then the
    function performs a blockwise inversion and returns the inverted A.
    """
    n = len(A)
    E, F, G, H = A[0:idx, 0:idx], A[0:idx, idx:n],  A[idx:n, 0:idx], A[idx:n, idx:n]  # E on nyt idx * idx -matriisi
    E = np.linalg.inv(E)
    F = -E@F
    H = H+G@F
    G, H = G@E, np.linalg.inv(H)
    G = -H@G
    # the line below can be parallelized if we can make sure
    # both operations take their own copy of the value of F as it is
    # before the line, and only after this F is overwritten by F@H
    E, F  = E+F@G, F@H
    return np.block([[E, F], [G, H]])

def block_iter_inv(M,i):
    """Function performs an iterative inversion of 2-D numpy array M in i>1 iterations by partitioning it
    into smaller blocks and inverting these one by one using a block Gaussian approach"""
    n = len(M)
    k = int(n/i)
    m = 2*k
    E, F, G, H = M[0:k, 0:k], M[0:k, k:m], M[k:m, 0:k], M[k:m, k:m] 
    E = block_inv(E, int(k/2))
    while m <= i*k:
        F = -E@F
        H = H+G@F
        G, H = G@E, block_inv(H, int(k/2))
        G = -H@G
        # the line below can be parallelized if we can make sure
        # both operations take their own copy of the value of F as it is
        # before the line, and only after this F is overwritten by F@H
        E, F = E+F@G, F@H
        if m < (i-1)*k:
            #Replace E with the inverted submatrix and form a block partition of a larger submatrix of M
            E, F, G, H = np.block([[E, F], [G, H]]), M[0:m, m:m+k], M[m:m+k, 0:m], M[m:m+k, m:m+k] 
        elif m == (i-1)*k: # Need to make sure that the last block partition doesn't "leave out" some cells as a result of rounding in k = int(n/i)
            E, F, G, H = np.block([[E, F], [G, H]]), M[0:m, m:n], M[m:n, 0:m], M[m:n, m:n]
        m = m+k     
    return np.block([[E, F], [G, H]])   

# # 41 x 41 -matriisi
# ds_1 = xr.open_dataset("data/regions_verify_isotope_202112_cov.nc")
# bio_1 = ds_1["covariance_bio"]

#  6858 x 6858 -matriisi
# ds_2 = xr.open_dataset("data/regions_verify_202104_cov.nc")
# bio_2 = ds_2["covariance_bio"]
# anth_2 = ds_2["covariance_anth"]

# bio_2 = bio_2.to_numpy()
# anth_2 = anth_2.to_numpy()

# inv1 = np.linalg.inv(bio_2)
# inv2 = block_inv(bio_2, 1000)
# inv3 = block_iter_inv(bio_2, 15)

# iv1, inv3 = xr.DataArray(inv1), xr.DataArray(inv3)

# # inv_diff = abs(inv1 - inv3)

# fig, ax = plt.subplots(layout  = 'constrained')
# inv3.plot.pcolormesh(yincrease = False, robust = True)
# ax.set_title('Bio_202104 block_iter_inv (15 iterations)')
# plt.show()


