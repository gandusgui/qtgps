import numpy as np

from .my_fft3D import my_fft3D
from .my_ifft3D import my_ifft3D


def apply_laplace_operator_fft(fsg, pw_in):

    fourpi = 4 * np.pi
    # prefactor = fourpi
    prefactor = 1.0
    average = 0.0

    Gsqrd = fsg.squared
    if fsg.singularity is not None:
        i, j, k = fsg.singularity
        Gsqrd[i, j, k] = average
    D = Gsqrd

    pw_in_hat = my_fft3D(pw_in, fsg)
    pw_out_hat = prefactor * D * pw_in_hat
    pw_out = my_ifft3D(pw_out_hat, fsg)

    pw_out = np.real(pw_out)
    return pw_out
