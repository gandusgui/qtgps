import numpy as np
from scipy.linalg import norm


def mollify(rsg, sig, M):

    x = rsg.x
    y = rsg.y
    z = rsg.z

    Lx = rsg.Lx
    Ly = rsg.Ly
    Lz = rsg.Lz

    G2 = np.zeros(rsg.size)
    sigsqr = sig ** 2
    sigma = sig

    X, Y, Z = rsg.get_grid()

    xsqr = X ** 2 + Y ** 2 + Z ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = (X - Lx) ** 2 + (Y - Ly) ** 2 + (Z - Lz) ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = X ** 2 + (Y - Ly) ** 2 + (Z - Lz) ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = (X - Lx) ** 2 + Y ** 2 + (Z - Lz) ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = (X - Lx) ** 2 + (Y - Ly) ** 2 + Z ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = X ** 2 + Y ** 2 + (Z - Lz) ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = (X - Lx) ** 2 + Y ** 2 + Z ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    xsqr = X ** 2 + (Y - Ly) ** 2 + Z ** 2
    cond = xsqr < sigma
    G2[cond] = np.exp(1 / (xsqr[cond] / sigsqr - 1))

    # for i,j,k in np.ndindex(*rsg.size):
    #     xi = x[i]   yj = y[j]   zk = z[k]
    #     if (norm([xi, yj, zk]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi, yj, zk],sig))**2 - 1))
    #     elif (norm([xi-Lx, yj-Ly, zk-Lz]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi-Lx, yj-Ly, zk-Lz],sig))**2 - 1))
    #     elif (norm([xi, yj-Ly, zk-Lz]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi, yj-Ly, zk-Lz],sig))**2 - 1))
    #     elif (norm([xi-Lx, yj, zk-Lz]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi-Lx, yj, zk-Lz],sig))**2 - 1))
    #     elif (norm([xi-Lx, yj-Ly, zk]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi-Lx, yj-Ly, zk],sig))**2 - 1))
    #     elif (norm([xi, yj, zk-Lz]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi, yj, zk-Lz],sig))**2 - 1))
    #     elif (norm([xi-Lx, yj, zk]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi-Lx, yj, zk],sig))**2 - 1))
    #     elif (norm([xi, yj-Ly, zk]) < sig):
    #         G2[i,j,k] = np.exp(1/(norm(np.divide([xi, yj-Ly, zk],sig))**2 - 1))

    return G2
    # G2 *= (1 / sig)**3
    # trpz = G2.sum()
    # G2 /= trpz
    #
    # cstr_hat = np.fft.fftn(M)
    # G_hat = np.fft.fftn(G2)
    # M_sig = np.fft.ifftn(cstr_hat*G_hat)
    # M_sig[M_sig <= 1e-10] = 0
    #
    # return M_sig
