import numpy as np

from .apply_laplace_operator_fft import apply_laplace_operator_fft
from .apply_P_operator import apply_P_operator


def apply_poisson_operator_fft(v, rsg, fsg, epsilon):

    Pxv = apply_P_operator(rsg, epsilon, v)
    ALxv = apply_laplace_operator_fft(fsg, v)
    Axv = ALxv + Pxv

    return Axv
