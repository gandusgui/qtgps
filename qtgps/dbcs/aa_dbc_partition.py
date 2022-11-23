import numpy as np
from .tools import ov2vertices

def aa_dbc_partition(dbc):

    from .aa_dbc_setup import Tile

    x_xtnt, y_xtnt, z_xtnt = dbc.get_extent()

    x_xtnt_p = np.linspace(x_xtnt[0], x_xtnt[1], dbc.ntiles[0]+1)
    y_xtnt_p = np.linspace(y_xtnt[0], y_xtnt[1], dbc.ntiles[1]+1)
    z_xtnt_p = np.linspace(z_xtnt[0], z_xtnt[1], dbc.ntiles[2]+1)

    for i,j,k in np.ndindex(*dbc.ntiles):
        o = np.array([x_xtnt_p[i], y_xtnt_p[j], z_xtnt_p[k]])
        v = [x_xtnt_p[i+1], y_xtnt_p[j+1], z_xtnt_p[k+1]] - o
        tile = Tile(dbc.vD[i,j,k], dbc.smoothing_width)
        ov2vertices(o, v, tile.vertices)
        dbc.tiles.append(tile)
