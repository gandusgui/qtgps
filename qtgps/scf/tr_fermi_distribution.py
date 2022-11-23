import numpy as np
# Physical quantities
from  scipy.constants import e, k

k_B = k / e # Boltzmann constant [eV/K] 8.6173303e-05

def fermidistribution(energies, mu=0., T=300):

    return 1. / (1. + np.exp((energies - mu) / (k_B*T)))
