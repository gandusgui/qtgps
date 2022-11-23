import numpy as np

from .apply_inv_laplace_operator_fft import apply_inv_laplace_operator_fft


def ps_implicit_prepare_blocks(dbcs, rsg, fsg):

    n = np.prod(rsg.size)
    p = 0  # number of tiles
    for dirichlet in dbcs:
        p += np.prod(dirichlet.ntiles)

    B = np.zeros((n, p))  # nxp
    Bt = np.zeros((p, n))  # pxn
    QAinvxB = np.zeros((n, p))  # nxn
    vD = np.zeros(p)

    count = 0
    for dirichlet in dbcs:
        for tile in dirichlet.tiles:
            pw_in = tile.pw / tile.volume
            Bt[count, :] = pw_in.flat
            QAinvxB[:, count] = apply_inv_laplace_operator_fft(fsg, pw_in).flat
            vD[count] = tile.vD
            count += 1

    B = Bt.T
    QS = -Bt @ QAinvxB  # pxp
    Bxunit_vec = Bt.sum(1) / n  # p

    R = np.zeros((p + 1, p + 1))
    R[:p, :p] = QS
    R[:p, p:] = Bxunit_vec.reshape(p, 1)
    R[p:, :p] = Bxunit_vec

    Rinv = np.linalg.inv(R)

    B.shape = tuple(rsg.size) + (p,)
    Bt.shape = (p,) + tuple(rsg.size)

    return B, Bt, QS, Rinv, vD
