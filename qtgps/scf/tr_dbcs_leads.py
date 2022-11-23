import numpy as np

from poisson import *

def get_dbcs_leads(rsg, atoms, naL, naR, V, dna, dnb, sigma=0.3, dir='x'):

    d = 'xyz'.index(dir)
    #Transverse directions
    t_dirs = [0,1,2]
    t_dirs.pop(d)
    n = rsg.size
    L = rsg.L

    LdL = atoms[naL+1].position[d] - atoms[0].position[d]
    LdR = atoms[-1].position[d] - atoms[-naR-1].position[d]

    print('Setting planes at {} and {}'.format(LdL, L[d]-LdR))

    ndL = int(LdL // rsg.dx)
    ndR = int((L[d] - LdR) // rsg.dx)

    VL = V[ndL]
    VR = V[ndR]

    #Dirichlet conditions
    VDL = VL[dna//2::dna, dnb//2::dnb]
    VDR = VR[dna//2::dna, dnb//2::dnb]

    shapeL = np.ones(3, dtype=int)
    shapeL[t_dirs] = VDL.shape
    shapeR = np.ones(3, dtype=int)
    shapeR[t_dirs] = VDR.shape

    c = rsg.delta[d]
    a, b = L[t_dirs]

    dbcL = PlanarDbc(rsg, VDL, sigma, shapeL, dir,
                     [ndL*rsg.dx-c/2,0,0], a, b, c)
    dbcR = PlanarDbc(rsg, VDR, sigma, shapeR, dir,
                     [(ndR-1)*rsg.dx-c/2,0,0], a, b, c)

    return dbcL, dbcR
