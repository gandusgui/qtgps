import numpy as np

from .my_fft3D import my_fft3D
from .my_ifft3D import my_ifft3D


def apply_inv_laplace_operator_fft(fsg, pw_in):

    fourpi = 4*np.pi
    # prefactor = 1.0/fourpi
    prefactor = 1.
    average = 0.

    Gsqrd = fsg.squared
    if fsg.singularity is not None:
        i, j, k = fsg.singularity
        Gsqrd[i,j,k] = np.inf
    D_inv = 1 / Gsqrd

    pw_in_hat = my_fft3D(pw_in, fsg)
    pw_out_hat = prefactor*D_inv*pw_in_hat
    if fsg.singularity is not None:
        pw_out_hat[i,j,k] = average
    pw_out = my_ifft3D(pw_out_hat, fsg)

    pw_out = np.real(pw_out)

    return pw_out
