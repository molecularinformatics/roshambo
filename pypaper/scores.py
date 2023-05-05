import numpy as np


def calc_tversky(ref_overlap, fit_overlap, ref_fit_overlap, alpha, beta):
    """
    Calculates the Tversky score between two sets of molecular characteristics
    (e.g. shape and color). The tversky score takes into account the relative
    importance of each molecule. It is asymmetric, but setting alpha = beta = 0.5 makes
    it symmetric.

    Args:
        ref_overlap (Union[int, numpy.ndarray]):
            The self overlap of the reference molecule.
        fit_overlap (Union[int, numpy.ndarray]):
            The self overlap of the fit molecule.
        ref_fit_overlap (Union[int, numpy.ndarray]):
            The overlap between the reference molecule and the fit molecule.
        alpha (float):
            A weighting factor that determines the importance of the reference features
            in the calculation of the Tversky index.
        beta (float):
            A weighting factor that determines the importance of the fit features
            in the calculation of the Tversky index.

    Returns:
        Union[float, np.ndarray]:
            The Tversky coefficient between the reference and fit molecules.
    """

    return ref_fit_overlap / (alpha * ref_overlap + beta * fit_overlap)


def calc_tanimoto(ref_overlap, fit_overlap, ref_fit_overlap):
    """
    Calculates the Tanimoto score between two sets of molecular characteristics
    (e.g. shape and color). The Tanimoto score is symmetric, with a value between 0.0
     and 1.0.

    Args:
        ref_overlap (Union[int, numpy.ndarray]):
            The self overlap of the reference molecule.
        fit_overlap (Union[int, numpy.ndarray]):
            The self overlap of the fit molecule.
        ref_fit_overlap (Union[int, numpy.ndarray]):
            The overlap between the reference molecule and the fit molecule.

    Returns:
        Union[float, np.ndarray]:
            The Tanimoto coefficient between the reference and fit molecules.
    """

    return ref_fit_overlap / (ref_overlap + fit_overlap - ref_fit_overlap)


def scores(outputs, ref_volume):
    """Calculate Tanimoto, Fit Tversky (alpha = 0.05 and beta = 0.95), and
    Reference Tversky(alpha = 0.95 and beta = 0.05) scores based on outputs
    and reference volume.

    Args:
        outputs (List[Tuple[float, float]]):
            List of tuples containing the outputs of the overlap calculations
            (shape or color). Each tuple should contain (fit_overlap, ref_fit_overlap).
        ref_volume (float):
            Volume of the reference molecule.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the following scores:
            - Tanimoto: Tanimoto score between the reference and fit overlaps.
            - Fit Tversky: Tversky score calculated between the reference and
              fit overlaps, with a higher weight given to the fit molecules.
            - Reference Tversky: Tversky score calculated between the reference
              and fit overlaps, with a higher weight given to the reference set.
            - Full Reference Fit Overlap: Overlap between the reference and fit molecules.
    """
    outputs_array = np.array(outputs)
    full_fit_overlap = outputs_array[:, 0]
    full_ref_fit_overlap = outputs_array[:, 1]
    full_ref_overlap = np.ones_like(full_fit_overlap) * ref_volume
    tanimoto = calc_tanimoto(full_ref_overlap, full_fit_overlap, full_ref_fit_overlap)
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
