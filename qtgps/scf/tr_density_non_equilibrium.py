import numpy as np

from .tr_fermi_distribution import fermidistribution
from .tr_mat_list import MatLists

def get_density_non_equilibrium(G, energies, mu1, mu2, T=300):

    f1 = fermidistribution(energies, mu1, T)
    f2 = fermidistribution(energies, mu2, T)
    dE = energies[1] - energies[0]

    gn = MatLists.zeros_like(G.hs_list_ii[0])
    temp = MatLists()
    for e, energy in enumerate(energies):
        temp << G.lesser(energy, f1[e], f2[e])
        gn += temp.real

    S = MatLists()
    S <<  (G.hs_list_ii[1], G.hs_list_ij[1], G.hs_list_ji[1])
    GS = gn @ S
    return GS.diagonal() / np.pi * dE
