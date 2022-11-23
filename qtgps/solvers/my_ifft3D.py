import numpy as np


def my_ifft3D(A_hat, fsg):

    if any(fsg.nbc) or any(fsg.dbc):

        # else
        nx = fsg.nx
        ny = fsg.ny
        nz = fsg.nz

        A = fsg.zeros()

        for i1, i2 in np.ndindex(nx, ny):
            func = fsg.get_ifft(dir=2)
            A[i1, i2, :] = func(A_hat[i1, i2, :])
        for i1, i2 in np.ndindex(nx, nz):
            func = fsg.get_ifft(dir=1)
            A[i1, :, i2] = func(A[i1, :, i2])
        for i1, i2 in np.ndindex(ny, nz):
            func = fsg.get_ifft(dir=0)
            A[:, i1, i2] = func(A[:, i1, i2])

    else:
        A = np.fft.ifftn(A_hat)

    return A
