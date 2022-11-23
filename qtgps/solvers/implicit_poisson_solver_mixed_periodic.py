import numpy as np

from .apply_inv_laplace_operator_fft import apply_inv_laplace_operator_fft
from .apply_P_operator import apply_P_operator
from .apply_poisson_operator_fft import apply_poisson_operator_fft
from .ps_implicit_compute_error_fft import ps_implicit_compute_error_fft
from .ps_implicit_prepare_blocks import ps_implicit_prepare_blocks


def implicit_poisson_solver_mixed_periodic(
    dbcs, rsg, fsg, epsilon, tol, omega, max_iter, density
):

    n = np.prod(rsg.size)

    def avg(X):
        return np.sum(X) / n

    p = 0
    for dirichlet in dbcs:
        p += np.prod(dirichlet.ntiles)
    sz = rsg.size

    print("preparing blocks ...")
    B, Bt, QS, Rinv, vD = ps_implicit_prepare_blocks(dbcs, rsg, fsg)
    print("Done!")

    lambda0 = np.zeros(p)
    w = np.zeros(p + 1)

    fourpi = 4 * np.pi
    g = fourpi * density / epsilon.eps
    g_avg = avg(g)

    lambda_old = lambda0

    v_old = rsg.zeros()
    tmp3D = apply_poisson_operator_fft(v_old, rsg, fsg, epsilon)
    Bxlambda_old = B @ lambda_old
    res_old = g - tmp3D - Bxlambda_old

    QAinvxres = apply_inv_laplace_operator_fft(fsg, res_old)

    reached_max_iter = False
    reached_tol = False

    iter = 1
    Bt.shape = (p, n)
    while (~reached_max_iter) & (~reached_tol):
        v_new = v_old + omega * QAinvxres
        Axvbar = apply_P_operator(rsg, epsilon, v_new)
        Axvbar_avg = avg(Axvbar)
        gminusAxvbar_avg = g_avg - Axvbar_avg

        QSxlambda = QS @ lambda_old
        Btxv_bar = Bt @ v_new.flat
        w[:p] = QSxlambda + vD - Btxv_bar
        w[-1] = gminusAxvbar_avg
        lambda_newNeta = Rinv @ w
        lambda_new = lambda_newNeta[:p]
        eta = lambda_newNeta[-1]

        v_new = v_new + eta / n

        Bxlambda_new = B @ lambda_new
        PxQAinvxres = apply_P_operator(rsg, epsilon, QAinvxres)
        res_new = (
            (1 - omega) * res_old - omega * PxQAinvxres + Bxlambda_old - Bxlambda_new
        )

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
        lambda_old = lambda_new
        Bxlambda_old = Bxlambda_new

    return v_new
