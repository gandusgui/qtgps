from functools import partial

import numpy as np
from scipy.fft import dct, dst, fft, ifft

from .tools import grid2vecs

"""Documentation:

https://arxiv.org/pdf/1409.8116.pdf
"""


def _get_xgrid(n, d, is_neumann, is_dirichlet):
    if is_neumann:
        return np.pi * np.arange(n) / (d * n)
    if is_dirichlet:
        return np.pi * np.arange(1, n + 1) / (d * n)
    else:
        return 2 * np.pi * np.fft.fftfreq(n, d)


class ReciprocalGrid:
    def __init__(self, rsg, neumann_directions, dirichlet_directions):

        nx = rsg.nx
        ny = rsg.ny
        nz = rsg.nz
        dx = rsg.dx
        dy = rsg.dy
        dz = rsg.dz
        # bases = rsg.bases
        self.nbc = np.zeros(3, dtype=bool)
        self.nbc[0] = "x" in neumann_directions
        self.nbc[1] = "y" in neumann_directions
        self.nbc[2] = "z" in neumann_directions

        self.dbc = np.zeros(3, dtype=bool)
        self.dbc[0] = "x" in dirichlet_directions
        self.dbc[1] = "y" in dirichlet_directions
        self.dbc[2] = "z" in dirichlet_directions

        self.kx = _get_xgrid(nx, dx, self.nbc[0], self.dbc[0])
        self.ky = _get_xgrid(ny, dy, self.nbc[1], self.dbc[1])
        self.kz = _get_xgrid(nz, dz, self.nbc[2], self.dbc[2])

        self.size = np.array([nx, ny, nz])

        if any(self.dbc):
            self.singularity = None
        else:
            self.singularity = np.array([0, 0, 0])
        self.squared = self._squared()

    def get_grid(self):
        kx = self.kx  # 2*pi/Lx
        ky = self.ky  # 2*pi/Ly
        kz = self.kz  # 2*pi/Lz
        kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing="ij")
        return kX, kY, kZ

    def get_kvec(self, T=False):
        kX, kY, kZ = self.get_grid()
        return grid2vecs(kX, kY, kZ, T)

    def get_fft(self, dir):
        if self.nbc[dir]:
            return partial(dct, type=2)
        elif self.dbc[dir]:
            return partial(dst, type=2)
        else:
            return fft

    def get_ifft(self, dir):
        if self.nbc[dir]:
            #
            def _partial(x):
                return 1 / (2 * n) * dct(x, type=3)

            n = self.size[dir]
            return _partial
        if self.dbc[dir]:
            #
            def _partial(x):
                return 1 / (2 * n) * dst(x, type=3)

            n = self.size[dir]
            return _partial
        else:
            return ifft

    def _squared(self):
        kX, kY, kZ = self.get_grid()
        return kX ** 2 + kY ** 2 + kZ ** 2

    # Helpers
    def zeros(self, dtype=complex):
        return np.zeros(self.size, dtype=dtype)

    @property
    def nx(self):
        return self.size[0]

    @property
    def ny(self):
        return self.size[1]

    @property
    def nz(self):
        return self.size[2]
