def calc_tversky(ref_overlap, fit_overlap, ref_fit_overlap, alpha, beta):
    return ref_fit_overlap / (alpha * ref_overlap + beta * fit_overlap)


def calc_tanimoto(ref_overlap, fit_overlap, ref_fit_overlap):
    return ref_fit_overlap / (ref_overlap + fit_overlap - ref_fit_overlap)

