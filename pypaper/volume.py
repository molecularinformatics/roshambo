import numpy as np

PARTIAL_ALPHA = 2.41798793102
PI = 3.14159265358


def calculate_analytic_overlap_volume(ref_mol, fit_mol):
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


def calculate_tanimoto_analytic(ref_mol, fit_mol, ref_overlap):
    fit_overlap = calculate_analytic_overlap_volume(fit_mol, fit_mol)
    ref_fit_overlap = calculate_analytic_overlap_volume(ref_mol, fit_mol)
    tanimoto = ref_fit_overlap / (ref_overlap + fit_overlap - ref_fit_overlap)
    return tanimoto
