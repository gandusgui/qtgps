import numpy as np

from .aa_dbc_partition import aa_dbc_partition
from .aa_tile_pw_compute import aa_tile_pw_compute


def dirichlet_bc_partition(rsg, dbc):

    aa_dbc_partition(dbc)

    if dbc.geom in ["planar", "cuboidal"]:
        for tile in dbc.tiles:
            tile.pw = aa_tile_pw_compute(rsg, tile)
            tile.volume = np.sum(tile.pw)

    else:
        raise NotImplementedError(
            "Dirichlet boundary condition {} not implemented".format(dbc.geom)
        )

    # case "cylindrical"
    #     tiles = arbitrary_dbc_partition(vertices, n_prtn(1:2))
    #     n_tiles = numel(tiles)
    #
    #     for k = 1:n_tiles
    #         tiles(k).tile.pw = zeros(numel(x_grid),numel(y_grid),numel(z_grid))
    #         tiles(k).tile.pw = arbitrary_tile_pw_compute(x_grid, y_grid, z_grid, tiles(k).tile.vertices, sigma)
    #         tiles(k).tile.volume = sum(tiles(k).tile.pw(:))
