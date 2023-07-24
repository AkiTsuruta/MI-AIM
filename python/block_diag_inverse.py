import numpy as np
import scipy.linalg
defaultinv = np.linalg.inv


def get_blocks(M, nblock):
    """Partition the area around the diagonal of matrix M into nblock blocks of
    approximately equal size."""
    n = len(M)
    blsize = n//nblock
    nrem = n%nblock #remainder
    asize = (nblock-nrem)*blsize
    a = np.arange(0, asize, blsize)
    b = np.arange(asize, n+blsize+1, blsize+1)
    coords = np.concatenate((a, b))
    blocks = [M[coords[i]:coords[i+1],coords[i]:coords[i+1]] for i in np.arange(len(coords)-1)]
    return blocks


def invert_combine(blocks):
    """Compute the inverses of 2D arrays given in an 1D array and
    combine the inverted arrays into a block diagonal array"""
    for i in np.arange(len(blocks)):
        blocks[i] = defaultinv(blocks[i])
    iM = scipy.linalg.block_diag(*blocks)
    return iM


def block_diag_inv(M, nblock=5):
    """Return a block diagonal approximation of the inverse of M. The inverse
    is computed by splitting the elements around the diagonal of M into nblock blocks,
    computing the inverses of these blocks and combining these to a block diagonal matrix"""
    blocks = get_blocks(M, nblock)
    iM = invert_combine(blocks)
    return iM
