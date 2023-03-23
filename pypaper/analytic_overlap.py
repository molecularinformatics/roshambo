import time
import math

import numpy as np

from itertools import combinations

KAPPA = 2.41798793102
PI = 3.14159265358
CONSTANT_P = (4 / 3) * PI * (KAPPA / PI) ** 1.5
EXP = math.exp(1)


def calc_analytic_overlap_vol(ref_mol, fit_mol):
    ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
    fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()

    ref_coords = ref_mol_coords_radii[:, :3]
    fit_coords = fit_mol_coords_radii[:, :3]

    ref_radii = ref_mol_coords_radii[:, 3]
    fit_radii = fit_mol_coords_radii[:, 3]

    dist_sqr = np.sum(
        (ref_coords[:, np.newaxis, :] - fit_coords[np.newaxis, :, :]) ** 2, axis=2
    )

    pi = (4 / 3) * PI * (KAPPA / PI) ** 1.5
    pij = pi * pi
    ai = (KAPPA / (ref_radii**2))[:, np.newaxis]
    aj = (KAPPA / (fit_radii**2))[np.newaxis, :]
    aij = ai + aj
    dij = ai * aj * dist_sqr
    vij = pij * np.exp(-dij / aij) * (PI / aij) ** 1.5
    overlap = np.sum(vij)
    return overlap


def calc_multi_analytic_overlap_vol(ref_mol, fit_mol):
    fit_overlap = calc_analytic_overlap_vol(fit_mol, fit_mol)
    ref_fit_overlap = calc_analytic_overlap_vol(ref_mol, fit_mol)
    return fit_overlap, ref_fit_overlap


def calc_single_overlap(atom_inds, alpha_dict, cross_alpha_distance_dict):
    p = CONSTANT_P ** (len(atom_inds))
    alpha = sum([alpha_dict[i] for i in atom_inds])
    k = [cross_alpha_distance_dict[i][j] for i, j in combinations(atom_inds, 2)]
    k_exp = EXP ** (-sum(k) / alpha)
    return p * k_exp * (PI / alpha) ** 1.5


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
    alpha_dict = {i: KAPPA / j[3] ** 2 for i, j in enumerate(all_radii)}
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
    alpha_dict = {i: KAPPA / j[3] ** 2 for i, j in enumerate(all_radii)}
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
