import numpy as np
from  scipy.special import erf

def aa_tile_pw_compute(rsg, dbc):

    x_xtnt, y_xtnt, z_xtnt = dbc.get_extent()
    sigma = dbc.smoothing_width

    x = rsg.x
    y = rsg.y
    z = rsg.z

    Pix = 0.5 * (erf((x-x_xtnt[0])/sigma) - erf((x-x_xtnt[1])/sigma))
    Piy = 0.5 * (erf((y-y_xtnt[0])/sigma) - erf((y-y_xtnt[1])/sigma))
    Piz = 0.5 * (erf((z-z_xtnt[0])/sigma) - erf((z-z_xtnt[1])/sigma))

    PiX, PiY, PiZ = np.meshgrid(Pix, Piy, Piz, indexing='ij')
    Pi = PiX * PiY * PiZ

    return Pi
