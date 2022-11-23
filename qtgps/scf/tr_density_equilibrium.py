import numpy as np
# Physical quantities
from scipy.constants import e, k

from .tr_mat_list import MatLists
from .tr_zero_fermi import zero_fermi

# from transport.solvers.recursive import *


def get_density_equilibrium(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1/(k_B*T)
    a_p = mu + 1j*zp/beta

    eta = G.eta
    G.eta = complex(0.)

    R = 1e10
    # \mu_0
    mu0 = MatLists()
    mu0 << G.retarded(1j*R)
    mu0 = mu0.imag

    mu1 = MatLists.zeros_like(mu0)
    temp = MatLists()
    for i in range(N):
        temp << G.retarded(a_p[i])
        mu1 += temp.real * (4 / beta * Rp[i])

    # Density matrix
    rho = mu0 * R + mu1
    rho *= -1

    G.eta = eta

    # n = \int Tr[GS]
    S = MatLists()
    S <<  (G.hs_list_ii[1], G.hs_list_ij[1], G.hs_list_ji[1])
    GS = rho @ S

    return GS.diagonal()
