import numpy as np


def apply_P_operator(rsg, epsilon, V):

    dx = rsg.dx
    dy = rsg.dy
    dz = rsg.dz

    grad_ln_epsx = epsilon.ln_deps_dx
    grad_ln_epsy = epsilon.ln_deps_dy
    grad_ln_epsz = epsilon.ln_deps_dz

    grad_Vx, grad_Vy, grad_Vz = np.gradient(V, dx, dy, dz)

    result = grad_ln_epsx * grad_Vx + grad_ln_epsy * grad_Vy + grad_ln_epsz * grad_Vz

    result = -result

    return result
