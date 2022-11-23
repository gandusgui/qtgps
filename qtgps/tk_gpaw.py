import numpy as np

from poisson import *

def get_bf_descriptor(calc):

    calc = calc
    nao = calc.wfs.setups.nao
    M_a = calc.wfs.setups.M_a

    bfd = BasisFunctionDescriptor(M_a, nao)

    return bfd

def get_poisson_solver(calc):

    calc = calc
    n = calc.wfs.gd.N_c
    L = calc.atoms.cell.diagonal()

    rsg = RealSpaceGrid(L, n)
    dielectric = DielectricGrid(rsg)

    poissonsolver = PoissonSolver()
    poissonsolver.set_grid_descriptor(rsg)
    poissonsolver.set_dielectric_descriptor(dielectric)

    return poissonsolver
