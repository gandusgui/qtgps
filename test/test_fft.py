import numpy as np
from ase.io import read
from poisson import *
from poisson.solvers.implicit_poisson_solver_periodic import (
    implicit_poisson_solver_periodic,
)

# from poisson.rs_grid import RealSpaceGrid
# from poisson.generate_dielectric import DielectricGrid


atoms = read("inputdata/scatt.xyz")
n = [184, 64, 76]
L = atoms.cell.diagonal()
rho = np.load("inputdata/rho.npy")

rsg = RealSpaceGrid(L, n)
fsg = rsg.get_reciprocal()

rho_elec = rsg.interpolate(atoms, rho)

dielectric = 1
epsilon = DielectricGrid(rsg)  # , rho_elec, dielectric)

max_iter = 30
omega = 1
tol = 1e-8

# % == periodic ===========================
V = implicit_poisson_solver_periodic(rsg, fsg, epsilon, tol, omega, max_iter, rho_elec)

expected = np.load("fft/V.npy")
np.testing.assert_allclose(V, expected)
