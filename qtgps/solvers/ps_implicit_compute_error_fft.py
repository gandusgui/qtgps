import numpy as np

from .apply_inv_laplace_operator_fft import apply_inv_laplace_operator_fft


def ps_implicit_compute_error_fft(rsg, fsg, res_new, v_old, v_new):

    vol = rsg.VoxelV

    # evaluate \Delta^-1(res) = \Delta^-1 (g - \Delta(v_new) - P(v_new) + Bt \lambda)
    QAinvxres_new = apply_inv_laplace_operator_fft(fsg, res_new)
    # (normalized) preconditioned residual norm error :
    pres_error = np.sqrt(np.sum(QAinvxres_new[:] ** 2)) / vol

    # normalized absolute error :
    # nabs_error := \frac{\| v_old - v_new \|}{volume}
    nabs_error = np.sum((np.abs(v_old - v_new) ** 2))
    nabs_error = np.sqrt(nabs_error) / vol

    return pres_error, nabs_error, QAinvxres_new
