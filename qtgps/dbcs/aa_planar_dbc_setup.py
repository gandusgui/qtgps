import numpy as np
from .aa_dbc_setup import DirichletBoundaryCondition as DBC
from .tools import ov2vertices

class PlanarDbc(DBC):
    """order is always 'xyz'.

        n = 'x' a ='y' b = 'z'
        n = 'y' a ='x' b = 'z'
        n = 'z' a ='x' b = 'y'
        """

    geom = 'planar'

    def __init__(self, rsg, vD, smoothing_width, ntiles, abs_normal, origin, a, b, c):


        self.out_plane = 'xyz'.index(abs_normal)
        self.in_plane = np.delete([0,1,2], self.out_plane)

        if isinstance(vD, (int,float)):
            vD = np.tile(vD, ntiles)
        elif vD.ndim == 2:
            vD = np.expand_dims(vD, self.out_plane)
        assert vD.shape[self.out_plane] == 1

        self.origin = origin
        self.a = a
        self.b = b
        self.c = c

        super().__init__(rsg, vD, smoothing_width, ntiles)

    def _init(self):
        # out_plane = 'xyz'.index(self.abs_normal)
        # in_plane = np.delete([0,1,2], out_plane)
        o = self.origin
        v = np.zeros(3)
        v[self.out_plane] = self.c
        v[self.in_plane] = [self.a, self.b]
        ov2vertices(o, v, self.vertices)
