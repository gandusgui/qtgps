from __future__ import print_function
import numpy as np
import pickle
from ase.io import read
from gpaw import restart
from transport.lcao.principallayer import PrincipalSelfEnergy
from transport.greenfunction import RecursiveGF
from matplotlib import pyplot as plt

#energies = np.arange(-2, 3, 0.01)
from poisson import *
from poisson.tk_gpaw import get_bf_descriptor, get_poisson_solver
from poisson.scf.tr_density_equilibrium import get_density_equilibrium
from poisson.scf.tr_screening_approximation import * # equilibrium_setup

atoms = read('inputdata/scatt.xyz')
patoms, pcalc = restart('./inputdata/leads.gpw',txt=None)
H_kMM, S_kMM = pickle.load(open('inputdata/hs_leads_k.pckl','rb'))
hs_list_ii, hs_list_ij = pickle.load(open('./inputdata/hs_scatt_lists.pckl','rb'))

eta = 1e-5

RSE = [PrincipalSelfEnergy(pcalc, scatt=atoms, id=0),
       PrincipalSelfEnergy(pcalc, scatt=atoms, id=1)]

for selfenergy in RSE:
    selfenergy.set(eta=eta)
    selfenergy.initialize(H_kMM, S_kMM)

RGF = RecursiveGF(selfenergies=RSE)

RGF.set(eta=eta, align_bf=0)
RGF.initialize(hs_list_ii, hs_list_ij)

nzp = 3
T = 100

density = get_density_equilibrium(RGF, nzp=nzp)
calc = restart('inputdata/scatt.gpw', txt=None)[1]
bfd = get_bf_descriptor(calc)
rho, weights = bfd.sum(density)

ps = get_poisson_solver(calc)

naL = naR = RSE[0].natoms
dbcL, dbcR, V_H = equilibrium_setup(ps, rho, atoms, naL, naR, dna=50, dnb=50)

dbcG = PlanarDbc(ps.rsg, -1, 0.3, [1,1,1], 'z', [13,0,3.5], 25.3-13, ps.rsg.L[1], 0.1)
dbcs = [dbcL, dbcR, dbcG]

scf = SCF()
scf.solve(bfd, RGF, ps, rho, weights, atoms, V_H, T, nzp, dbcs=dbcs, neumann_directions='x')
