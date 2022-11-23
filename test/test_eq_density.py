import pickle
import numpy as np
from time import time

from ase.io import read
from gpaw import restart

from transport.lcao import principallayer as pl
from transport import greenfunction as gf
from poisson.scf.tr_density_equilibrium import get_density_equilibrium

# h, s = pickle.load(open('./inputdata/hs_scat.pckl','rb'))
# satoms, scalc = restart('./inputdata/scatt.gpw',txt=None)
atoms = read('inputdata/scatt.xyz')
patoms, pcalc = restart('./inputdata/leads.gpw',txt=None)
H_kMM, S_kMM = pickle.load(open('inputdata/hs_leads_k.pckl','rb'))
hs_list_ii, hs_list_ij = pickle.load(open('./inputdata/hs_scatt_lists.pckl','rb'))

eta = 1e-5

RSE = [pl.PrincipalSelfEnergy(pcalc, scatt=atoms, id=0),
       pl.PrincipalSelfEnergy(pcalc, scatt=atoms, id=1)]

for selfenergy in RSE:
    selfenergy.set(eta=eta)
    selfenergy.initialize(H_kMM, S_kMM)

RGF = gf.RecursiveGF(selfenergies=RSE)

RGF.set(eta=eta, align_bf=0)
RGF.initialize(hs_list_ii, hs_list_ij)

t = time()
ans = get_density_equilibrium(RGF, nzp=3).sum()
print(time() - t)
expected = 6620.575134322985
assert np.allclose(ans, expected)
