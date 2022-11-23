import numpy as np
from numba import njit

@njit
def mollifier(x, y, z, sig):#, M):

    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    Lz = z[-1] - z[0]

    nx = len(x) ny = len(y) nz = len(z)

    G2 = np.zeros((ny,ny,nz))

    for i,j,k in np.ndindex(nx,ny,nz):
        xi = x[i]   yj = y[j]   zk = z[k]
        if (np.linalg.norm(np.array([xi, yj, zk])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi, yj, zk]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi-Lx, yj-Ly, zk-Lz])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi-Lx, yj-Ly, zk-Lz]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi, yj-Ly, zk-Lz])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi, yj-Ly, zk-Lz]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi-Lx, yj, zk-Lz])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi-Lx, yj, zk-Lz]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi-Lx, yj-Ly, zk])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi-Lx, yj-Ly, zk]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi, yj, zk-Lz])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi, yj, zk-Lz]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi-Lx, yj, zk])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi-Lx, yj, zk]),sig)**2 - 1))
        elif (np.linalg.norm(np.array([xi, yj-Ly, zk])) < sig):
            G2[i,j,k] = np.exp(1/(np.linalg.norm(np.array([xi, yj-Ly, zk]),sig)**2 - 1))

    G2 *= (1 / sig)**3
    trpz = G2.sum()
    G2 /= trpz

    return G2
    # cstr_hat = np.fft.fftn(M)
    # G_hat = np.fft.fftn(G)
    # M_sig = np.fft.ifftn(cstr_hat*G_hat)
    # M_sig[M_sig <= 1e-10] = 0
    #
    # return M_sig
