import numpy as np

from pypaper.grid import Grid


PARTIAL_ALPHA = 2.41798793102
PI = 3.14159265358


def calc_analytic_overlap_vol(ref_mol, fit_mol):
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()

    ref_coords = ref_mol_coords_radii[:, :3]
    fit_coords = fit_mol_coords_radii[:, :3]

    ref_radii = ref_mol_coords_radii[:, 3]
    fit_radii = fit_mol_coords_radii[:, 3]

    ref_alpha = PARTIAL_ALPHA / (ref_radii**2)
    fit_alpha = PARTIAL_ALPHA / (fit_radii**2)

    dist_sqr = np.sum(
        (ref_coords[:, np.newaxis, :] - fit_coords[np.newaxis, :, :]) ** 2, axis=2
    )

    ref_alpha_matrix = ref_alpha[:, np.newaxis]
    fit_alpha_matrix = fit_alpha[np.newaxis, :]
    sum_alpha = ref_alpha_matrix + fit_alpha_matrix
    k = np.exp(-ref_alpha_matrix * fit_alpha_matrix * dist_sqr / sum_alpha)
    v = 8 * k * (PI / sum_alpha) ** 1.5
    overlap = np.sum(v)
    return overlap


def calc_multi_analytic_overlap_vol(ref_mol, fit_mol):
    fit_overlap = calc_analytic_overlap_vol(fit_mol, fit_mol)
    ref_fit_overlap = calc_analytic_overlap_vol(ref_mol, fit_mol)
    return fit_overlap, ref_fit_overlap


def rho(atoms, gcs):
    rt22 = 2.82842712475
    alphas = -PARTIAL_ALPHA / (atoms[:, 0, 3] ** 2)
    diffs = gcs[:, np.newaxis, :] - atoms[:, 0, :3]
    r2s = np.sum(diffs * diffs, axis=-1)
    rhos = rt22 * np.exp(alphas[np.newaxis, :] * r2s)
    return rhos


def calc_gaussian_overlap_vol(ref_mol, fit_mol, grid):
    gcs = grid.converted_grid
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()

    rho_ref = rho(ref_mol_coords_radii[:, np.newaxis], gcs)
    ref_grid = 1 - np.prod(1 - rho_ref, axis=1)

    rho_fit = rho(fit_mol_coords_radii[:, np.newaxis], gcs)
    fit_grid = 1 - np.prod(1 - rho_fit, axis=1)
    volume = np.sum(ref_grid * fit_grid) * grid.res**3
    return volume


def calc_multi_gaussian_overlap_vol(fit_mol, res, margin, ref_grid, ref_mol):
    fit_grid = Grid(fit_mol, res=res, margin=margin)
    fit_grid.create_grid()
    fit_overlap = calc_gaussian_overlap_vol(fit_mol, fit_mol, fit_grid)

    ref_fit_overlap = calc_gaussian_overlap_vol(
        ref_mol,
        fit_mol,
        ref_grid if np.prod(ref_grid.extent) < np.prod(fit_grid.extent) else fit_grid,
    )
    return fit_overlap, ref_fit_overlap
