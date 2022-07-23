import numpy as np
invm = np.linalg.inv

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
    return np.block([[E, F], [np.transpose(F), H]])


