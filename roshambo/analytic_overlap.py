import time

import numpy as np

from itertools import combinations

from roshambo import constants


def calc_analytic_overlap_vol(ref_mol, fit_mol):
    """
    Calculates the Gaussian overlap volume between two molecules using an analytic
    method. Please refer to Eq. (11) and (12) of this paper by Grant et. al.:
    https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K.

    Args:
        ref_mol (Molecule):
            The reference molecule object.
        fit_mol (Molecule):
            The fit molecule object.

    Returns:
        float:
            The analytic overlap volume between the two molecules.
    """

    # Get the atomic coordinates and radii for each molecule
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()

    # Get the atomic coordinates for each molecule
    ref_coords = ref_mol_coords_radii[:, :3]
    fit_coords = fit_mol_coords_radii[:, :3]

    # Get the atomic radii for each molecule
    ref_radii = ref_mol_coords_radii[:, 3]
    fit_radii = fit_mol_coords_radii[:, 3]

    # Calculate the distance squared between each pair of atoms
    dist_sqr = np.sum(
        (ref_coords[:, np.newaxis, :] - fit_coords[np.newaxis, :, :]) ** 2, axis=2
    )

    # Calculate the constants needed for the Gaussian overlap volume calculation
    pij = constants.CONSTANT_P * constants.CONSTANT_P
    ai = (constants.KAPPA / (ref_radii**2))[:, np.newaxis]
    aj = (constants.KAPPA / (fit_radii**2))[np.newaxis, :]
    aij = ai + aj
    dij = ai * aj * dist_sqr

    # Calculate the Gaussian overlap volume for each pair of atoms
    vij = pij * np.exp(-dij / aij) * (constants.PI / aij) ** 1.5

    # Sum up the Gaussian overlap volumes to get the total overlap volume
    overlap = np.sum(vij)
    return overlap


def calc_multi_analytic_overlap_vol(ref_mol, fit_mol):
    """
    Calculates the self-overlap volume of the fit molecule and the overlap volume
    between a reference molecule and a fit molecule using the calc_analytic_overlap_vol
    function.

    Args:
        ref_mol (Molecule):
            The reference molecule object.
        fit_mol (Molecule):
            The fit molecule object.

    Returns:
        Tuple[float, float]:
            A tuple containing the following:
            - Fit overlap: The self overlap volume of the fit molecule.
            - Reference fit overlap: The overlap volume between the reference molecule
            and the fit molecule.
    """
    fit_overlap = calc_analytic_overlap_vol(fit_mol, fit_mol)
    ref_fit_overlap = calc_analytic_overlap_vol(ref_mol, fit_mol)
    return fit_overlap, ref_fit_overlap


def calc_single_overlap(atom_inds, alpha_dict, cross_alpha_distance_dict):
    """
    Uses the Gaussian product theorem to calculate the product of n Gaussians, as a
    single Gaussian function.

    Args:
        atom_inds (list):
            List of atomic indices for which to calculate the single Gaussian function.
        alpha_dict (dict):
            Dictionary mapping atomic indices to alpha values (KAPPA/radius**2).
        cross_alpha_distance_dict (dict):
            Dictionary of dictionaries of this form:
            {i: {j: alpha_i * alpha_j * (dist_ij)**2, k: ...}, ...}, where
            i, j, k, ... are atomic indices.

    Returns:
        float:
            The calculated Gaussian function.
    """

    # Calculate the product of the Gaussian weight factors
    p = constants.CONSTANT_P ** (len(atom_inds))

    # Calculate the total exponent as the sum of the alpha values
    alpha = sum([alpha_dict[i] for i in atom_inds])

    k = [cross_alpha_distance_dict[i][j] for i, j in combinations(atom_inds, 2)]
    # Calculate the exponent term and return the final overlap value
    k_exp = constants.EXP ** (-sum(k) / alpha)
    return p * k_exp * (constants.PI / alpha) ** 1.5


def _calc_overlap(
    atom_inds,
    rem_ind,
    len_full,
    len_ref,
    n,
    alpha_dict,
    cross_alpha_distance_dict,
    cross_distance_bool_dict,
):
    """
    Calculates the overlap between two molecules based on the gaussian product theorem.

    Args:
        atom_inds (list):
            List of atom indices for which the overlap needs to be calculated.
        rem_ind (set):
            Set of atom indices which should be skipped.
        len_full (int):
            Number of atoms in the fit molecule.
        len_ref (int):
            Number of atoms in the reference molecule.
        n (int):
            The order of the overlap calculation.
        alpha_dict (dict):
            Dictionary containing alpha values for each atom index.
        cross_alpha_distance_dict (dict):
            Dictionary containing pairwise alpha distance values.
        cross_distance_bool_dict (dict):
            Dictionary containing boolean values for pairwise distance values between
            the reference and fit molecules.

    Returns:
        float: The overlap between the two molecules.
    """
    if n == 1:
        return calc_single_overlap(atom_inds, alpha_dict, cross_alpha_distance_dict)
    else:
        main_overlap = (
            _calc_overlap(
                atom_inds,
                rem_ind,
                len_full,
                len_ref,
                1,
                alpha_dict,
                cross_alpha_distance_dict,
                cross_distance_bool_dict,
            )
            if len(atom_inds) > 1
            else 0
        )

        num_of_itrs = len_ref if len(atom_inds) == 1 else len_full
        iter_list = [i for i in range(num_of_itrs) if i not in rem_ind]
        higher_order = 0
        for ind in iter_list:
            flag = np.prod([cross_distance_bool_dict[(ind, i)] for i in atom_inds])
            if flag:
                higher_order += _calc_overlap(
                    atom_inds + [ind],
                    rem_ind | {ii for ii in range(ind + 1)},
                    len_full,
                    len_ref,
                    n - 1,
                    alpha_dict,
                    cross_alpha_distance_dict,
                    cross_distance_bool_dict,
                )
        return main_overlap - higher_order


def calc_analytic_overlap_vol_recursive(
    ref_mol, fit_mol, n=6, proxy_cutoff=None, epsilon=0.1, use_carbon_radii=False
):
    # TODO: add a restriction on n to be even
    # TODO: add a restriction on n to be at max, the smaller of the sizes of the two molecules
    # TODO: change carbon_radii calculations to be much faster
    # TODO: precompute alpha values based on atomic radii
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii(
        use_carbon_radii=use_carbon_radii
    )
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii(
        use_carbon_radii=use_carbon_radii
    )
    ref_len = len(ref_mol_coords_radii)
    all_radii = np.concatenate([ref_mol_coords_radii, fit_mol_coords_radii])
    alpha_dict = {i: constants.KAPPA / j[3] ** 2 for i, j in enumerate(all_radii)}
    cross_alpha_dict = {
        (i, j): alpha_dict[i] * alpha_dict[j]
        for i in range(len(all_radii))
        for j in range(len(all_radii))
    }
    distance_dict = {
        (i, j): np.linalg.norm(all_radii[i][:3] - all_radii[j][:3])
        for i in range(len(all_radii))
        for j in range(len(all_radii))
    }
    if proxy_cutoff:
        cross_distance_bool_dict = {
            (i, j): distance_dict[(i, j)] <= proxy_cutoff
            for i in range(len(all_radii))
            for j in range(len(all_radii))
        }
    else:
        cross_distance_bool_dict = {
            (i, j): distance_dict[(i, j)] <= all_radii[i][3] + all_radii[j][3] + epsilon
            for i in range(len(all_radii))
            for j in range(len(all_radii))
        }

    cross_alpha_distance_dict = {
        i: {
            j: cross_alpha_dict[(i, j)] * distance_dict[(i, j)] ** 2
            for j in range(len(all_radii))
        }
        for i in range(len(all_radii))
    }
    overlaps = 0
    for ind in range(len(fit_mol_coords_radii)):
        overlaps -= _calc_overlap(
            [ind + len(ref_mol_coords_radii)],
            {len(ref_mol_coords_radii) + k for k in range(ind + 1)},
            len(alpha_dict),
            ref_len,
            n,
            alpha_dict,
            cross_alpha_distance_dict,
            cross_distance_bool_dict,
        )
    return overlaps


def calc_multi_analytic_overlap_vol_recursive(
    ref_mol, fit_mol, n=2, proxy_cutoff=None, epsilon=0.1, use_carbon_radii=False
):
    fit_overlap = calc_analytic_overlap_vol_recursive(
        fit_mol,
        fit_mol,
        n=n,
        proxy_cutoff=proxy_cutoff,
        epsilon=epsilon,
        use_carbon_radii=use_carbon_radii,
    )
    ref_fit_overlap = calc_analytic_overlap_vol_recursive(
        ref_mol,
        fit_mol,
        n=n,
        proxy_cutoff=proxy_cutoff,
        epsilon=epsilon,
        use_carbon_radii=use_carbon_radii,
    )
    return fit_overlap, ref_fit_overlap


def calc_analytic_overlap_vol_iterative(ref_mol, fit_mol, n=6):
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()
    ref_len = len(ref_mol_coords_radii)
    all_radii = np.concatenate([ref_mol_coords_radii, fit_mol_coords_radii])
    alpha_dict = {i: constants.KAPPA / j[3] ** 2 for i, j in enumerate(all_radii)}
    cross_alpha_dict = {
        (i, j): alpha_dict[i] * alpha_dict[j]
        for i in range(len(all_radii))
        for j in range(len(all_radii))
    }
    cross_alpha_distance_dict = {
        (i, j): cross_alpha_dict[(i, j)]
        * np.sum((all_radii[i][:3] - all_radii[j][:3]) ** 2)
        for i in range(len(all_radii))
        for j in range(len(all_radii))
    }

    full_len = len(all_radii)
    overlap = 0
    tot_time = 0
    for fit_atom_ind in range(len(fit_mol_coords_radii)):
        remove_2 = {len(ref_mol_coords_radii) + k for k in range(fit_atom_ind + 1)}
        for ref_atom_ind in range(len(ref_mol_coords_radii)):
            atom_inds = [fit_atom_ind + ref_len, ref_atom_ind]
            overlap += calc_single_overlap(
                atom_inds, alpha_dict, cross_alpha_distance_dict
            )
            if n == 2:
                continue
            remove_3 = remove_2 | {ii for ii in range(ref_atom_ind + 1)}
            for third_order_ind in range(full_len):
                if third_order_ind in remove_3:
                    continue
                third_order_atom_inds = atom_inds + [third_order_ind]
                overlap -= calc_single_overlap(
                    third_order_atom_inds, alpha_dict, cross_alpha_distance_dict
                )
                if n == 3:
                    continue
                remove_4 = remove_3 | {ii for ii in range(third_order_ind + 1)}
                for fourth_order_ind in range(full_len):
                    if fourth_order_ind in remove_4:
                        continue
                    fourth_order_atom_inds = third_order_atom_inds + [fourth_order_ind]
                    overlap += calc_single_overlap(
                        fourth_order_atom_inds, alpha_dict, cross_alpha_distance_dict
                    )
                    if n == 4:
                        continue
                    remove_5 = remove_4 | {ii for ii in range(fourth_order_ind + 1)}
                    for fifth_order_ind in range(full_len):
                        if fifth_order_ind in remove_5:
                            continue
                        fifth_order_atom_inds = fourth_order_atom_inds + [
                            fifth_order_ind
                        ]
                        st = time.time()
                        overlap -= calc_single_overlap(
                            fifth_order_atom_inds, alpha_dict, cross_alpha_distance_dict
                        )
                        tot_time += time.time() - st
                        if n == 5:
                            continue
                        remove_6 = remove_5 | {ii for ii in range(fifth_order_ind + 1)}
                        for sixth_order_ind in range(full_len):
                            if sixth_order_ind in remove_6:
                                continue
                            sixth_order_atom_inds = fifth_order_atom_inds + [
                                sixth_order_ind
                            ]
                            overlap += calc_single_overlap(
                                sixth_order_atom_inds,
                                alpha_dict,
                                cross_alpha_distance_dict,
                            )
    print("TOT TIME", tot_time)
    return overlap
