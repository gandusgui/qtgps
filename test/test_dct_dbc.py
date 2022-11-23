import numpy as np
from ase.io import read
from poisson import *
from poisson.solvers.implicit_poisson_solver_mixed_periodic import (
    implicit_poisson_solver_mixed_periodic,
)

# from poisson.rs_grid import RealSpaceGrid
# from poisson.generate_dielectric import DielectricGrid
# from poisson.aa_planar_dbc_setup import PlanarDbc


atoms = read("inputdata/scatt.xyz")
n = [184, 64, 76]
L = atoms.cell.diagonal()
rho = np.load("inputdata/rho.npy")

dirichlet_directions = "z"
rsg = RealSpaceGrid(L, n)
fsg = rsg.get_reciprocal(dirichlet_directions=dirichlet_directions)

rho_elec = rsg.interpolate(atoms, rho)

dielectric = 1
epsilon = DielectricGrid(rsg)  # , rho_elec, dielectric)

max_iter = 30
omega = 1
tol = 1e-8

# % == non-periodic ===========================
dbcs = []

vD = 1.0
origin = [0.0, 0.0, 5]
a = L[0] / 3
b = L[1]
c = 2
sig = 0.5
dbcs.append(PlanarDbc(rsg, vD, sig, [2, 2, 1], "z", origin, a, b, c))

vD = 1.0
origin = [L[0] * 2 / 3, 0.0, 5]
a = L[0] / 3
b = L[1]
c = 2
sig = 0.5
dbcs.append(PlanarDbc(rsg, vD, sig, [2, 2, 1], "z", origin, a, b, c))

vD = 2
origin = [16.5, 2.0, 10]
a = 5
b = 3
c = 0.1
sig = 0.5
dbcs.append(PlanarDbc(rsg, vD, sig, [2, 1, 1], "z", origin, a, b, c))

V = implicit_poisson_solver_mixed_periodic(
    dbcs, rsg, fsg, epsilon, tol, omega, max_iter, rho_elec
)

# np.save('dct_dbc/V.npy', V)
# from ase.io import write
# write('V.cube', atoms, data=V)
expected = np.load("dct_dbc/V.npy")
np.testing.assert_allclose(V, expected)
