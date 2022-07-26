import numpy as np
invm = np.linalg.inv
block = np.block
transp = np.transpose

"""A version of the block_inv and block_iter_inv functions
that utilizes symmetry of the covariance matrix to be inverted.

Need to test if this is faster than the one that doesn't take symmetry into account"""


def block_inv_s(M, idx):
    n = len(M)
    E = M[0:idx, 0:idx]
    E = invm(E)
    F = M[0:idx, idx:n]
    G = M[idx:n, 0:idx]
    H = M[idx:n, idx:n]  
    F = -E@F
    H = invm(H+G@F)
    F = F@H
    E = E-F@G@E
    return block([[E, F], [transp(F), H]])

def block_iter_inv_s(M,i):
    """Function performs an iterative inversion of 2-D numpy array M in i>1 iterations by partitioning it
    into smaller blocks and inverting these one by one using a block Gaussian approach
    
    Utilizes symmetry & np.transpose"""
    n = len(M)
    k = int(n/i)
    l = k+k
    E = M[0:k, 0:k]   
    E = invm(E)
    F = M[0:k, k:l]
    G = M[k:l, 0:k]
    H = M[k:l, k:l]
    for m in range(l, (i+1)*k, k):
        F = -E@F
        H = invm(H+G@F)
        F = F@H
        E = E-F@G@E
        if m < (i-1)*k:
            #Replace E with the inverted submatrix and form a block partition of a larger submatrix of M
            E = block([[E, F], [transp(F), H]])
            F = M[0:m, m:m+k]
            G = M[m:m+k, 0:m]
            H = M[m:m+k, m:m+k] 
        elif m == (i-1)*k: # Need to make sure that the last block partition doesn't "leave out" some cells as a result of rounding in k = int(n/i)
            E = block([[E, F], [transp(F), H]])
            F = M[0:m, m:n]
            G = M[m:n, 0:m]
            H = M[m:n, m:n]    
    return block([[E, F], [transp(F), H]])


def block_iter_inv_s2(M,i):
    """Function performs an iterative inversion of 2-D numpy array M in i>1 iterations by partitioning it
    into smaller blocks and inverting these one by one using a block Gaussian approach
    
    Utilizes symmetry, doesn't compute np.transpose"""
    n = len(M)
    k = int(n/i)
    l = k+k
    E = M[0:k, 0:k]   
    E = invm(E)
    F = M[0:k, k:l]
    G = M[k:l, 0:k]
    H = M[k:l, k:l]
    for m in range(l, (i+1)*k, k):
        F = -E@F
        H = invm(H+G@F)
        G = -H@G@E
        E = E+F@G
        F = F@H
        if m < (i-1)*k:
            #Replace E with the inverted submatrix and form a block partition of a larger submatrix of M
            E = block([[E, F], [G, H]])
            F = M[0:m, m:m+k]
            G = M[m:m+k, 0:m]
            H = M[m:m+k, m:m+k] 
        elif m == (i-1)*k: # Need to make sure that the last block partition doesn't "leave out" some cells as a result of rounding in k = int(n/i)
            E = block([[E, F], [G, H]])
            F = M[0:m, m:n]
            G = M[m:n, 0:m]
            H = M[m:n, m:n]    
    return block([[E, F], [G, H]])

