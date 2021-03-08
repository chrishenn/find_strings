
import numpy as np
from numba import njit, prange



# @njit(parallel=True)
@njit
def frnn_cpu(pts, imgid, lin_radius):

    size = pts.shape[0]
    loops = size*size
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)

    for i in prange( loops ):

        ptid0 = int( np.divide( i, size ) )
        ptid1 = int( np.mod( i, size ) )

        p0 = pts[ptid0, :2]
        p1 = pts[ptid1, :2]

        imgid0 = imgid[ptid0]
        imgid1 = imgid[ptid1]

        if imgid0 == imgid1 and ptid0 > ptid1:

            diff_yx = p0 - p1
            dist_yx = np.sqrt( np.sum( np.power(diff_yx, 2) ) )

            if dist_yx < lin_radius:
                edges[i] = np.array( (ptid0, ptid1) )

    edges = edges[edges[:, 0] + edges[:, 1] > 0]
    return edges