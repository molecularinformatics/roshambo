import numpy as np


class Grid:
    def __init__(self, mol, res=0.4, margin=0.4):
        self.mol = mol
        self.res = res
        self.margin = margin
        self.extent = None
        self.converted_grid = None

    def get_bounding_box(self):
        coords_radii = self.mol.get_atomic_coordinates_and_radii()
        coords = coords_radii[:, :3]
        radii = coords_radii[:, 3]
        lb = coords - radii[:, None]
        ub = coords + radii[:, None]
        lb = lb.min(axis=0)
        ub = ub.max(axis=0)
        lb -= self.margin
        ub += self.margin
        return lb, ub

    def convert_grid_to_real(self, lb, coords):
        grid_coords = np.array(coords) + 0.5
        real_coords = lb + grid_coords * self.res
        return real_coords

    def create_grid(self):
        lb, ub = self.get_bounding_box()
        dims = ub - lb
        self.extent = np.ceil(dims / self.res).astype(int)
        self.converted_grid = np.array(
            [
                self.convert_grid_to_real(lb, coords)
                for coords in np.ndindex(self.extent[0], self.extent[1], self.extent[2])
            ]
        )
