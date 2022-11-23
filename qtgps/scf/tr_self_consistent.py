import numpy as np

from .tr_density_equilibrium import get_density_equilibrium
from .tr_density_non_equilibrium import get_density_non_equilibrium
from .tr_fermi_distribution import k_B

def mix_potential(V_H, V1, dV_Hinp_0, naL, naR, mu1=0., mu2=0., alpha=0.5):

    # Delta Hartree at step n
    dV_Hout_0 = V1 - V_H

    # Delta Hartree at step n+1
    dV_Hinp_1 = (1-alpha)*dV_Hinp_0 + alpha*dV_Hout_0
    # dV_Hinp_1[:naL] = mu1
    # dV_Hinp_1[-naR:] = mu2

    return dV_Hinp_1


def step_density(bfd, gf, dV_H, weights, dE=1e-3, mu1=0, mu2=0, T=300, nzp=100):

    dV_H_i = bfd.expand(dV_H, weights)
    gf.add_screening(dV_H_i)

    if mu1 == mu2:
        # Equilibrium
        density = get_density_equilibrium(gf, mu=mu1, T=T, nzp=nzp)
    if mu1 != mu2:
        # Out of equilibrium
        raise NotImplementedError('Out of equilibrium not implemented')
        # mu_min = min(mu1,mu2) - k_B*T
        # mu_max = max(mu1,mu2) + k_B*T
        # energies = np.arange(mu_min, mu_max, dE)
        # # \int_{-oo}^{mu_min}
        # mu = max(mu1,mu2)
        # gf.selfenergies[0].set_bias(mu)
        # gf.selfenergies[1].set_bias(mu)
        # density = get_density_equilibrium(gf, mu=mu_min, T=T, nzp=nzp)
        # # \int_{mu_min}^{mu_max}
        # gf.selfenergies[0].set_bias(mu1)
        # gf.selfenergies[1].set_bias(mu2)
        # density += get_density_non_equilibrium(gf, energies, mu1=mu1, mu2=mu2, T=T)

    rho, weights = bfd.sum(density)

    return rho, weights


def step_potential(ps, rho, atoms, **ps_kwargs):

    rsg = ps.rsg

    rho_ijk = rsg.interpolate(atoms, rho)
    V = ps.solve(rho_ijk, **ps_kwargs)
    V_a = rsg.integrate(atoms, V)

    return V_a
