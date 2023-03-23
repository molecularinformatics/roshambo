import numpy as np

from pypaper.grid import Grid

KAPPA = 2.41798793102
PI = 3.14159265358
RT22 = 2.82842712475


def rho(atoms, gcs):
    alphas = -KAPPA / (atoms[:, 0, 3] ** 2)
    diffs = gcs[:, np.newaxis, :] - atoms[:, 0, :3]
    r2s = np.sum(diffs * diffs, axis=-1)
    rhos = RT22 * np.exp(alphas[np.newaxis, :] * r2s)
    return rhos


def calc_gaussian_overlap_vol(ref_mol, fit_mol, grid, use_carbon_radii):
    gcs = grid.converted_grid
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii(use_carbon_radii)
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii(use_carbon_radii)

    rho_ref = rho(ref_mol_coords_radii[:, np.newaxis], gcs)
    ref_grid = 1 - np.prod(1 - rho_ref, axis=1)

    rho_fit = rho(fit_mol_coords_radii[:, np.newaxis], gcs)
    fit_grid = 1 - np.prod(1 - rho_fit, axis=1)
    volume = np.sum(ref_grid * fit_grid) * grid.res**3
    return volume


def calc_multi_gaussian_overlap_vol(fit_mol, res, margin, ref_grid, ref_mol, use_carbon_radii):
    fit_grid = Grid(fit_mol, res=res, margin=margin, use_carbon_radii=use_carbon_radii)
    fit_grid.create_grid()
    fit_overlap = calc_gaussian_overlap_vol(fit_mol, fit_mol, fit_grid, use_carbon_radii)

    ref_fit_overlap = calc_gaussian_overlap_vol(
        ref_mol,
        fit_mol,
        ref_grid if np.prod(ref_grid.extent) < np.prod(fit_grid.extent) else fit_grid,
        use_carbon_radii
    )
    return fit_overlap, ref_fit_overlap

