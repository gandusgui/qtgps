import numpy as np

from .apply_inv_laplace_operator_fft import apply_inv_laplace_operator_fft
from .apply_P_operator import apply_P_operator
from .apply_poisson_operator_fft import apply_poisson_operator_fft
from .ps_implicit_compute_error_fft import ps_implicit_compute_error_fft


def implicit_poisson_solver_periodic(rsg, fsg, epsilon, tol, omega, max_iter, density):

    fourpi = 4 * np.pi
    g = fourpi * density / epsilon.eps

    v_old = rsg.zeros()
    tmp3D = apply_poisson_operator_fft(v_old, rsg, fsg, epsilon)
    res_old = g - tmp3D

    QAinvxres = apply_inv_laplace_operator_fft(fsg, res_old)

    reached_max_iter = False
    reached_tol = False

    iter = 1
    while (~reached_max_iter) & (~reached_tol):
        v_new = v_old + omega * QAinvxres
        PxQAinvxres = apply_P_operator(rsg, epsilon, QAinvxres)
        res_new = (1 - omega) * res_old - omega * PxQAinvxres

        pres_error, nabs_error, QAinvxres_new = ps_implicit_compute_error_fft(
            rsg, fsg, res_new, v_old, v_new
        )
        QAinvxres = QAinvxres_new

        print(
            "iter: ",
            iter,
            "    pres_error: ",
            np.round(pres_error, 4),
            "    nabs_error: ",
            np.round(nabs_error, 4),
        )

        iter += 1
        reached_max_iter = iter > max_iter
        reached_tol = pres_error <= tol
        v_old = v_new
        res_old = res_new

    return v_new
