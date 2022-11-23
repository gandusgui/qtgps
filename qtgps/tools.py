import numpy as np

def grid2vecs(X, Y, Z, T=False):
    vecs = np.vstack((X.flat, Y.flat, Z.flat))
    if T is False:
        return vecs # (3,-1)
    return vecs.T # (-1,3)

