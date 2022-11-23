import numpy as np


def my_fft3D(A, fsg):

    if any(fsg.nbc) or any(fsg.dbc):

        # else
        nx = fsg.nx
        ny = fsg.ny
        nz = fsg.nz

        A_hat = fsg.zeros()

        for i1, i2 in np.ndindex(nx, ny):
            func = fsg.get_fft(dir=2)
            A_hat[i1, i2, :] = func(A[i1, i2, :])
        for i1, i2 in np.ndindex(nx, nz):
            func = fsg.get_fft(dir=1)
            A_hat[i1, :, i2] = func(A_hat[i1, :, i2])
        for i1, i2 in np.ndindex(ny, nz):
            func = fsg.get_fft(dir=0)
            A_hat[:, i1, i2] = func(A_hat[:, i1, i2])

    else:
        A_hat = np.fft.fftn(A)

    return A_hat
