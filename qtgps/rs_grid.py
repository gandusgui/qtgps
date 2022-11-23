import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .fs_grid import ReciprocalGrid
from .tools import grid2vecs


class RealSpaceGrid:
    def __init__(self, L, n):

        Lx, Ly, Lz = L
        nx, ny, nz = n

        self.L = np.array([Lx, Ly, Lz])

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.z = np.linspace(0, Lz, nz)

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        self.delta = np.array([dx, dy, dz])

        self.size = np.array([nx, ny, nz])

        ax = [dx, 0, 0]
        ay = [0, dy, 0]
        az = [0, 0, dz]
        self.bases = np.array([ax, ay, az]).T

        dV = np.dot(ax, np.cross(ay, az))
        self.VoxelV = np.prod(self.size) * dV  # np.dot(ax,np.cross(ay,az))

    def _get_bases(self, index):
        a = self.bases[:, index]
        a.shape = (3, 1)
        return a

    @property
    def ax(self):
        return self._get_bases(0)

    @property
    def ay(self):
        return self._get_bases(1)

    @property
    def az(self):
        return self._get_bases(2)

    def get_grid(self):
        x = self.x
        y = self.y
        z = self.z
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z

    def get_vec(self, T=False):
        X, Y, Z = self.get_grid()
        return grid2vecs(X, Y, Z, T)

    # Modifiers

    def get_reciprocal(self, neumann_directions=None, dirichlet_directions=None):
        if neumann_directions is None:
            neumann_directions = "-"
        if dirichlet_directions is None:
            dirichlet_directions = "-"
        return ReciprocalGrid(self, neumann_directions, dirichlet_directions)

    def interpolate(self, atoms, rho):

        indx = np.argmin(
            np.abs(atoms.positions[:, 0][:, None] - self.x[None, :]), axis=1
        )
        indy = np.argmin(
            np.abs(atoms.positions[:, 1][:, None] - self.y[None, :]), axis=1
        )
        indz = np.argmin(
            np.abs(atoms.positions[:, 2][:, None] - self.z[None, :]), axis=1
        )

        rho_elec = self.zeros()
        rho_elec[indx, indy, indz] = rho

        return rho_elec

    def integrate(self, atoms, V):
        interpolator = RegularGridInterpolator((self.x, self.y, self.z), V)
        return interpolator(atoms.positions)

    # Helpers
    def zeros(self, dtype=float):
        return np.zeros(self.size, dtype=dtype)

    def empty(self, dtype=float):
        return np.empty(self.size, dtype=dtype)

    @property
    def dx(self):
        return self.delta[0]

    @property
    def dy(self):
        return self.delta[1]

    @property
    def dz(self):
        return self.delta[2]

    @property
    def nx(self):
        return self.size[0]

    @property
    def ny(self):
        return self.size[1]

    @property
    def nz(self):
        return self.size[2]

    @property
    def Lx(self):
        return self.L[0]

    @property
    def Ly(self):
        return self.L[1]

    @property
    def Lz(self):
        return self.L[2]
