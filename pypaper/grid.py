import numpy as np


class Grid:
    """
    A class representing a grid for a molecule.

    Attributes:
        mol (Molecule):
            A Molecule object representing the molecule.
        res (float):
            The resolution of the grid.
        margin (float):
            The margin to add to the bounding box of the molecule when creating the grid.
        extent (np.ndarray):
            A 1D numpy array representing the dimensions of the grid.
        converted_grid (np.ndarray):
            A 2D numpy array representing the real coordinates of each point in the grid.
        lb (np.ndarray):
            A 1D numpy array representing the lower bounds of the grid.
        use_carbon_radii (bool):
            A boolean indicating whether to use carbon radii for the atoms.

    Args:
        mol (Molecule):
            A Molecule object representing the molecule.
        res (float):
            The resolution of the grid.
        margin (float):
            The margin to add to the bounding box of the molecule when creating the grid.
        use_carbon_radii (bool):
            A boolean indicating whether to use carbon radii for the atoms.
    """

    def __init__(self, mol, res, margin, use_carbon_radii):
        self.mol = mol
        self.res = res
        self.margin = margin
        self.extent = None
        self.converted_grid = None
        self.lb = None
        self.use_carbon_radii = use_carbon_radii

    def get_bounding_box(self):
        """
        Calculates the bounding box of the molecule.

        Returns:
            np.ndarray:
                A 1D numpy array representing the lower bounds of the bounding box.
            np.ndarray:
                A 1D numpy array representing the upper bounds of the bounding box.
        """
        coords_radii = self.mol.get_atomic_coordinates_and_radii(self.use_carbon_radii)
        coords = coords_radii[:, :3]
        radii = coords_radii[:, 3]
        lb = coords - radii[:, None]
        ub = coords + radii[:, None]
        lb = lb.min(axis=0)
        ub = ub.max(axis=0)
        lb -= self.margin
        ub += self.margin
        self.lb = lb
        return lb, ub

    def convert_grid_to_real(self, lb, coords):
        """
        Converts the coordinates of a point in the grid to real coordinates.

        Args:
            lb (np.ndarray):
                A 1D numpy array representing the lower bounds of the grid.
            coords (tuple):
                A tuple of integers representing the coordinates of the point in the grid.

        Returns:
            np.ndarray:
                A 1D numpy array representing the real coordinates of the point.
        """
        grid_coords = np.array(coords) + 0.5
        real_coords = lb + grid_coords * self.res
        return real_coords

    def create_grid(self):
        """
        Creates the grid.
        """
        lb, ub = self.get_bounding_box()
        dims = ub - lb
        self.extent = np.ceil(dims / self.res).astype(int)
        self.converted_grid = np.array(
            [
                self.convert_grid_to_real(lb, coords)
                for coords in np.ndindex(self.extent[0], self.extent[1], self.extent[2])
            ]
        )
