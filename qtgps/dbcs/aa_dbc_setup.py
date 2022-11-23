from abc import ABC, abstractmethod

import numpy as np

from .dirichlet_bc_partition import dirichlet_bc_partition


class Tile:
    def __init__(self, vD, smoothing_width):
        self.vD = vD
        self.smoothing_width = smoothing_width
        # Internal
        self.vertices = np.zeros((8, 3))
        self.pw = None
        self.volume = 0.0

    def get_extent(self):

        x_xtnt = [None for _ in range(2)]
        y_xtnt = [None for _ in range(2)]
        z_xtnt = [None for _ in range(2)]
        x_xtnt[0] = min(self.vertices[:, 0])
        x_xtnt[1] = max(self.vertices[:, 0])
        y_xtnt[0] = min(self.vertices[:, 1])
        y_xtnt[1] = max(self.vertices[:, 1])
        z_xtnt[0] = min(self.vertices[:, 2])
        z_xtnt[1] = max(self.vertices[:, 2])

        return x_xtnt, y_xtnt, z_xtnt


class DirichletBoundaryCondition(Tile, ABC):

    geom = "ABC"

    def __init__(self, rsg, vD, smoothing_width, ntiles):
        assert np.allclose(vD.shape, ntiles)
        self.ntiles = ntiles
        self.tiles = []
        self.geom = type(self).geom
        # Init variables
        Tile.__init__(self, vD, smoothing_width)
        ABC.__init__(self)
        # Compute vertices
        self._init()
        # Compute tiles
        dirichlet_bc_partition(rsg, self)

    @abstractmethod
    def _init(self, *args):
        pass

    def set_bias(self, bias):
        if not hasattr(self, "bias"):
            self.bias = 0.0
        if self.bias != 0.0:
            self._remove_bias(self.bias)
        self.bias = float(bias)
        for tile in self.tiles:
            tile.vD += bias

    def _remove_bias(self, bias):
        for tile in self.tiles:
            tile.vD -= bias
        self.bias = 0.0
