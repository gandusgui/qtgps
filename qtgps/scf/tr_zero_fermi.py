import numpy as np
import scipy.linalg as la

"""References:

http://www1.spms.ntu.edu.sg/~ydchong/teaching/08_contour_integration.pdf
"""


def zero_fermi(nzp):
    """Compute poles (zp) and residues (Rp) of fermi function."""
    N = nzp
    M = 2 * N

    A = np.zeros((2 * M, 2 * M))
    B = np.zeros((2 * M, 2 * M))

    zp = np.zeros(2 + M)
    Rp = np.zeros(2 + M)

    for i in range(1, M + 1):
        B[i, i] = 2 * i - 1

    for i in range(1, M):
        A[i, i + 1] = -0.5
        A[i + 1, i] = -0.5

    a = np.zeros(M * M)
    b = np.zeros(M * M)

    for i in range(M):
        for j in range(M):
            a[j * M + i] = A[i + 1, j + 1]
            b[j * M + i] = B[i + 1, j + 1]

    a.shape = (M, M)
    b.shape = (M, M)

    eigvas, eigvecs = la.eigh(a, b)

    zp[:M] = eigvas

    for i in range(M, 0, -1):
        zp[i] = zp[i - 1]

    for i in range(1, M + 1):
        zp[i] = 1.0 / zp[i]

    a = eigvecs.T.flatten()

    for i in range(0, M):
        Rp[i + 1] = -a[i * M] * a[i * M] * zp[i + 1] * zp[i + 1] * 0.250

    zp = -zp[1 : N + 1]
    Rp = Rp[1 : N + 1]

    return zp, Rp
