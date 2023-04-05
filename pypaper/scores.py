import numpy as np


def calc_tversky(ref_overlap, fit_overlap, ref_fit_overlap, alpha, beta):
    return ref_fit_overlap / (alpha * ref_overlap + beta * fit_overlap)


def calc_tanimoto(ref_overlap, fit_overlap, ref_fit_overlap):
    return ref_fit_overlap / (ref_overlap + fit_overlap - ref_fit_overlap)


def scores(outputs, ref_volume):
    outputs_array = np.array(outputs)
    full_fit_overlap = outputs_array[:, 0]
    full_ref_fit_overlap = outputs_array[:, 1]
    full_ref_overlap = np.ones_like(full_fit_overlap) * ref_volume
    tanimoto = calc_tanimoto(
        full_ref_overlap, full_fit_overlap, full_ref_fit_overlap
    )
    fit_tversky = calc_tversky(
        full_ref_overlap,
        full_fit_overlap,
        full_ref_fit_overlap,
        alpha=0.05,
        beta=0.95,
    )
    ref_tversky = calc_tversky(
        full_ref_overlap,
        full_fit_overlap,
        full_ref_fit_overlap,
        alpha=0.95,
        beta=0.05,
    )
    return tanimoto, fit_tversky, ref_tversky, full_ref_fit_overlap