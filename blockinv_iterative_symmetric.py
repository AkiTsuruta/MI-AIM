import numpy as np
invm = np.linalg.inv
block = np.block
transp = np.transpose

"""A version of the block_inv and block_iter_inv functions
that utilizes symmetry of the covariance matrix to be inverted.

Need to test if this is faster than the one that doesn't take symmetry into account"""


def block_inv_s(M, idx):
    n = len(M)
    E, F, G, H = M[0:idx, 0:idx], M[0:idx, idx:n],  M[idx:n, 0:idx], M[idx:n, idx:n]  # E on nyt idx * idx -matriisi
    E = invm(E)
    F = -E@F
    H = H+G@F
    H = invm(H)
    F = F@H
    E = E-F@G@E
    return block([[E, F], [transp(F), H]])


def block_iter_inv_s(M, i):
     """Function performs an iterative inversion of 2-D numpy array M in i>1 iterations by partitioning it
    into smaller blocks and inverting these one by one using a block Gaussian approach
    
    Utilizes symmetry and for-loop"""
    n = len(M)
    k = int(n/i)
    l = k+k
    E, F, G, H = M[0:k, 0:k], M[0:k, k:l], M[k:l, 0:k], M[k:l, k:l] 
    E = block_inv_s(E, int(k/2))
    for m in range(l, (i+1)*k, k):
        F = -E@F
        H = H+G@F
        G, H = G@E, block_inv_s(H, int(k/2))
        G = -H@G
        # the line below can be parallelized if we can make sure
        # both operations take their own copy of the value of F as it is
        # before the line, and only after this F is overwritten by F@H
        E, F = E+F@G, F@H
        if m < (i-1)*k:
            #Replace E with the inverted submatrix and form a block partition of a larger submatrix of M
            E, F, G, H = block([[E, F], [G, H]]), M[0:m, m:m+k], M[m:m+k, 0:m], M[m:m+k, m:m+k] 
        elif m == (i-1)*k: # Need to make sure that the last block partition doesn't "leave out" some cells as a result of rounding in k = int(n/i)
            E, F, G, H = block([[E, F], [G, H]]), M[0:m, m:n], M[m:n, 0:m], M[m:n, m:n]    
    return block([[E, F], [G, H]])
