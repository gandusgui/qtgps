import numpy as np
from .aa_dbc_setup import DirichletBoundaryCondition as DBC


class CuboidalDbc(DBC):

    geom = 'cuboidal'

    def __init__(self, rsg, vD, smoothing_width, ntiles, x_xtnt, y_xtnt, z_xtnt):

        if isinstance(vD, int):
            vD = np.tile(vD, ntiles)

        self.x_xtnt = x_xtnt
        self.y_xtnt = y_xtnt
        self.z_xtnt = z_xtnt

        super().__init__(rsg, vD, smoothing_width, ntiles)

    def _init(self):

        x_xtnt = self.x_xtnt
        y_xtnt = self.y_xtnt
        z_xtnt = self.z_xtnt

        o = np.array([x_xtnt[0], y_xtnt[0], z_xtnt[0]])
        v = [x_xtnt[1], y_xtnt[1], z_xtnt[1]] - o

        for i, z in enumerate([o[2], o[2]+v[2]]):
            self.vertices[0+i*4,:] = o[0], o[1], z
            self.vertices[1+i*4,:] = o[0]+v[0], o[1], z
            self.vertices[2+i*4,:] = o[0]+v[0], o[1]+v[1], z
            self.vertices[3+i*4,:] = o[0], o[1]+v[1], z
