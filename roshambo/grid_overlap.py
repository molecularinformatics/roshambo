import numpy as np

from roshambo import constants
from roshambo.grid import Grid


def rho(atoms, gcs):
    """
    Calculates the Gaussian density function using the atomic coordinates and radii
    and the given grid points. For more details about the function used, please refer
    to Eq. (1) of this paper:
    https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K.

    Args:
        atoms (np.ndarray):
            Array of atomic coordinates and radii.
        gcs (np.ndarray):
            Array of grid coordinates.

    Returns:
        np.ndarray:
            The values of the Gaussian density function.
    """

    # Calculate the alpha values using the atomic radii
    alphas = -constants.KAPPA / (atoms[:, 0, 3] ** 2)
    # Calculate the differences between the grid points and the atom coordinates
    diffs = gcs[:, np.newaxis, :] - atoms[:, 0, :3]
    # Calculate the r^2 values
    r2s = np.sum(diffs * diffs, axis=-1)
    # Calculate the rhos values
    rhos = constants.CONSTANT_P * np.exp(alphas[np.newaxis, :] * r2s)
    return rhos


def calc_gaussian_overlap_vol(ref_mol, fit_mol, grid, use_carbon_radii):
    """
    Calculates the overlap volume between two molecules using numerical integration by
    quadrature over a grid of a specific resolution. The integration is performed for
    Eq. (10) of this paper by Grant et. al.:
    https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K.

    Args:
        ref_mol (Molecule):
            The reference molecule object.
        fit_mol (Molecule):
            The fit molecule object.
        grid (Grid):
            The grid object.
        use_carbon_radii (bool):
            Whether to use carbon radii for the atoms or not.

    Returns:
        float:
            The overlap volume between the two molecules.
    """

    # Convert the coordinates of the points in the grid to real coordinates
    gcs = grid.converted_grid

    # Get the atomic coordinates and radii of the reference molecule
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii(use_carbon_radii)
    # Get the atomic coordinates and radii of the fitted molecule
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii(use_carbon_radii)

    # Calculate the Gaussian density function for the reference molecule
    rho_ref = rho(ref_mol_coords_radii[:, np.newaxis], gcs)
    ref_grid = 1 - np.prod(1 - rho_ref, axis=1)

    # Calculate the Gaussian density function for the fit molecule
    rho_fit = rho(fit_mol_coords_radii[:, np.newaxis], gcs)
    fit_grid = 1 - np.prod(1 - rho_fit, axis=1)

    # Calculate the overlap volume
    volume = np.sum(ref_grid * fit_grid) * grid.res**3
    return volume


def calc_multi_gaussian_overlap_vol(
    fit_mol, res, margin, ref_grid, ref_mol, use_carbon_radii
):
    """
    Calculates the self-overlap volume of the fit molecule and the overlap volume
    between a reference molecule and a fit molecule.

    Args:
        fit_mol (Molecule):
            The fit molecule object.
        res (float):
            The resolution of the grid.
        margin (float):
            The margin to add to the bounding box of the molecule when creating the grid.
        ref_grid (Grid):
            The grid object created using the reference molecule.
        ref_mol (Molecule):
            The reference molecule object.
        use_carbon_radii (bool):
            Whether to use carbon radii for the atoms or not.

    Returns:
        Tuple[float, float]:
            A tuple containing the following scores:
            - Fit overlap: The self overlap volume of the fit molecule.
            - Reference fit overlap: The overlap volume between the reference molecule
            and the fit molecule.
    """

    # Create the grid for the fit molecule
    fit_grid = Grid(fit_mol, res=res, margin=margin, use_carbon_radii=use_carbon_radii)
    fit_grid.create_grid()

    # Calculate the overlap volume of the fit molecule with itself
    fit_overlap = calc_gaussian_overlap_vol(
        fit_mol, fit_mol, fit_grid, use_carbon_radii
    )

    # Calculate the overlap volume of the reference molecule with the fit molecule
    ref_fit_overlap = calc_gaussian_overlap_vol(
        ref_mol,
        fit_mol,
        ref_grid if np.prod(ref_grid.extent) < np.prod(fit_grid.extent) else fit_grid,
        use_carbon_radii,
    )
    return fit_overlap, ref_fit_overlap
