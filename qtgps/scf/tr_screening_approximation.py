import numpy as np
from .tr_dbcs_leads import get_dbcs_leads
from .tr_self_consistent import *

def equilibrium_setup(ps, rho, atoms, naL, naR, dbcs=None, dna=10, dnb=10, sigma=0.3, dir='x'):

    rsg = ps.rsg

    rho_ijk = rsg.interpolate(atoms, rho)
    V = ps.solve(rho_ijk, dbcs=dbcs)
    dbcL, dbcR = get_dbcs_leads(rsg, atoms, naL, naR,
                                V, dna, dnb, sigma, dir)

    dbcs = [dbcL, dbcR] + (dbcs or [])
    V_H = ps.solve(rho_ijk, dbcs=dbcs)#, neumann_directions=dir)

    return dbcL, dbcR, rsg.integrate(atoms, V_H)


class SCF:

    def __init__(self, tol=0.1, max_niter=10, alpha=0.5):
        self.tol = tol
        self.alpha = alpha
        self.max_niter = max_niter

    def solve(self, bfd, gf, ps, rho, weights, atoms, V_H, T, nzp, **kwargs_ps):

        tol = self.tol
        alpha = self.alpha
        max_niter = self.max_niter

        dV_H = np.zeros_like(V_H)
        naL = gf.selfenergies[0].natoms
        naR = gf.selfenergies[1].natoms

        dV_list = []
        rho_list = [rho]
        iter = 0
        epsdV = np.inf
        while (abs(epsdV) > tol) & (iter < max_niter):

            # New potential
            V1 = step_potential(ps, rho, atoms, **kwargs_ps)

            # New dV_H
            dV1_H = mix_potential(V_H, V1, dV_H, naL, naR, alpha=alpha)
            # Screening approximation
            dV1_H[:naL] = 0.
            dV1_H[-naR:] = 0.

            # New charge
            rho1, weights1 = step_density(bfd, gf, dV1_H, weights, T=T, nzp=nzp)

            # Conv.
            epsdV = np.sum(dV1_H - dV_H)
            epsrho = np.sum(rho1 - rho)
            print(epsdV, epsrho)

            dV_list.append(dV1_H)
            rho_list.append(rho1)

            dV_H = dV1_H
            rho = rho1
            weights = weights1

            iter += 1

            dV = bfd.expand(dV_H, weights)
            np.save('dV', dV)

        np.save('conv_dV', dV_list)
        np.save('conv_rho', rho_list)
