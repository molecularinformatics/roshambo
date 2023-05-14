import os
import logging

import numpy as np

from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from roshambo import constants


FEATURES = {
    "Donor": [1.0, True],
    "Acceptor": [1.0, True],
    "PosIonizable": [1.0, False],
    "NegIonizable": [1.0, False],
    "Aromatic": [0.7, True],
    "Hydrophobe": [1.0, False],
}

def calc_pharm(rdkit_mol, fdef_path=None, write_to_file=False):
    """
    Calculates the pharmacophore features of an RDKit molecule.

    Args:
        rdkit_mol (rdkit.Chem.rdchem.Mol):
            The RDKit molecule for which the pharmacophore features will be calculated.
        fdef_path (str, optional):
            The path to the .fdef file. Defaults to None.
            Uses BaseFeatures.fdef if not provided. Note that this requires the
            RDKIT_DATA_DIR environment variable to be set.

    Returns:
        list:
            A list of pharmacophore features. Each feature is represented by a list
            containing the following elements:
            - The feature family name, e.g. Donor, Acceptor, PosIonizable, etc.
            - A list of atom indices corresponding to the feature.
            - A list of x, y, z coordinates for the feature.
            - The feature sigma value, i.e. radius.
            - A boolean indicating if the feature has a normal vector.
            - A list of x, y, z coordinates for the normal vector of the feature,
            if it exists.
    """

    if not fdef_path:
        # Get the path to the BaseFeatures.fdef file
        rdkit_data_path = os.environ.get("RDKIT_DATA_DIR")
        if rdkit_data_path:
            fdef_path = os.path.join(rdkit_data_path, "BaseFeatures.fdef")
        else:
            logging.error("RDKIT_DATA_DIR environment variable is not set.")
            return None

    # Build a feature factory using the .fdef file
    feature_factory = AllChem.BuildFeatureFactory(fdef_path)

    # Get a list of molecular features for the molecule using the feature factory
    features = feature_factory.GetFeaturesForMol(rdkit_mol)

    # Create an empty list to store the pharmacophore features
    pharmacophore = []
    for feature in features:
        fam = feature.GetFamily()
        if fam in FEATURES.keys():
            pos = feature.GetPos()
            atom_indices = feature.GetAtomIds()
            feature_data = FEATURES[fam]
            p = [
                fam,
                atom_indices,
                [pos[0], pos[1], pos[2]],
                feature_data[0],
                feature_data[1],
            ]
            # Calculate the normal vector for the feature, if it exists
            if feature_data[1]:
                if fam == "Aromatic":
                    n = calc_normal_aromatic(rdkit_mol, list(atom_indices), pos)
                else:
                    n = calc_normal(rdkit_mol, rdkit_mol.GetAtoms()[atom_indices[0]])
                p.append(n)
            pharmacophore.append(p)
    return pharmacophore


def calc_single_pharm_overlap(p1, p2):
    """
    Calculates the overlap between two pharmacophoric features using Gaussian density
    representation.

    Args:
        p1 (list):
            List of the first pharmacophoric feature in the following format -
            [family, atom_indices, [x, y, z], radius, directional]
        p2 (list):
            List of the second pharmacophoric feature in the following format -
            [family, atom_indices, [x, y, z], radius, directional]

    Returns:
        float:
            The overlap between the two pharmacophoric features.
    """

    # Calculate the distance between the two pharmacophoric features
    r2 = (
        (p1[2][0] - p2[2][0]) ** 2
        + (p1[2][1] - p2[2][1]) ** 2
        + (p1[2][2] - p2[2][2]) ** 2
    )

    # Calculate the overlap volume of the two pharmacophoric features
    vol = (constants.CONSTANT_P**2) * pow(constants.PI / (p1[3] + p2[3]), 1.5)
    vol *= np.exp(-(p1[3] * p2[3]) * r2 / (p1[3] + p2[3]))
    return vol


def calc_pharm_overlap(ref_pharm, fit_pharm):
    """
    Calculates the overlap between the pharmacophoric features of two molecules:
    reference and fit.

    Args:
        ref_pharm (list):
            A list of pharmacophoric features of the reference molecule.
        fit_pharm (list):
            A list of pharmacophoric features of the fit molecule.

    Returns:
        float: The pharmacophoric overlap volume between the reference and fit molecule.
    """

    volume = 0
    # Loop through each pair of pharmacophores from the reference and fit molecules
    for i, p1 in enumerate(ref_pharm):
        for j, p2 in enumerate(fit_pharm):
            # Check if the pharmacophore types match
            if p1[0] == p2[0]:
                # Calculate the volume overlap between the two pharmacophores
                vol = calc_single_pharm_overlap(p1, p2)
                # Add the volume overlap to the total volume
                volume += vol
    return volume


# def calc_pharm_overlap(ref_pharm, fit_pharm):
#     volume = 0
#     for i, p1 in enumerate(ref_pharm):
#         vols = []
#         for j, p2 in enumerate(fit_pharm):
#             if p1[0] == p2[0]:
#                 vol = calc_single_pharm_overlap(p1, p2)
#                 vols.append(vol)
#                 # print(p1[0], vol)
#         if vols:
#             volume += np.max(vols)
#     return volume


def calc_multi_pharm_overlap(fit_mol, ref_pharm, fdef_path):
    """
    Calculate the pharmacophoric overlap between the reference molecule and the fit
    molecule as well as the self-overlap of the fit molecule.

    Args:
        fit_mol (Molecule):
            A Molecule object representing the fit molecule.
        ref_pharm (list):
            A list of pharmacophoric features of the reference molecule.
        fdef_path (str):
            The path to the feature definition file.

    Returns:
        Tuple[float, float]:
            A tuple of two floats representing the pharmacophoric overlap of the fit
            molecule with itself and between the fit and reference molecules.
    """

    # Calculate the pharmacophore features for the fit molecule
    fit_pharm = calc_pharm(fit_mol.mol, fdef_path)
    # Calculate the self-overlap of the fit molecule
    fit_overlap = calc_pharm_overlap(fit_pharm, fit_pharm)
    # Calculate the pharmacophoric overlap between the reference molecule and the
    # fit molecule
    ref_fit_overlap = calc_pharm_overlap(ref_pharm, fit_pharm)
    return fit_overlap, ref_fit_overlap


def color_tanimoto(ref_mol, fit_mol, fdef_path=None):
    """
    Calculates the color Tanimoto similarity score between two molecules based on their
    pharmacophoric features.

    Args:
        ref_mol (rdkit.Chem.rdchem.Mol):
            RDKit molecule object for the reference molecule.
        fit_mol (rdkit.Chem.rdchem.Mol):
            RDKit molecule object for the fit molecule.
        fdef_path (str, optional):
            The path to the .fdef file. Defaults to None.
            Uses BaseFeatures.fdef if not provided.

    Returns:
        float:
            The color Tanimoto similarity score between the reference molecule and
            the fit molecule.
    """

    # Calculate the pharmacophoric properties of the reference and fit molecules
    ref_pharm = calc_pharm(ref_mol, fdef_path)
    fit_pharm = calc_pharm(fit_mol, fdef_path)
    ref_volume = 0
    fit_volume = 0
    overlap = 0

    # Calculate the volume of the reference molecule
    for i, p1 in enumerate(ref_pharm):
        for j, p2 in enumerate(ref_pharm):
            if p1[0] == p2[0]:
                vol = calc_single_pharm_overlap(p1, p2)
                ref_volume += vol

    # Calculate the volume of the fit molecule
    for i, p1 in enumerate(fit_pharm):
        for j, p2 in enumerate(fit_pharm):
            if p1[0] == p2[0]:
                vol = calc_single_pharm_overlap(p1, p2)
                fit_volume += vol

    # Calculate the overlapping volume of the reference and fit molecules
    for p1 in ref_pharm:
        for p2 in fit_pharm:
            if p1[0] == p2[0]:
                v = calc_single_pharm_overlap(p1, p2)
                overlap += v

    # Calculate the color Tanimoto similarity score between the reference and
    # fit molecules
    tanimoto = overlap / (ref_volume + fit_volume - overlap)
    return tanimoto


def calc_normal(rdkit_mol, atom):
    """
    Calculates the normal vector to the plane defined by the atom and its
    neighboring atoms.

    Args:
        rdkit_mol (rdkit.Chem.rdchem.Mol):
            An RDKit molecule object.
        atom (rdkit.Chem.rdchem.Atom):
            RDKit atom.

    Returns:
        list:
            The normal vector [x, y, z].
    """

    # Get the conformer of the molecule
    conf = rdkit_mol.GetConformer()
    # Get the 3D coordinates of the given atom
    atom_coords = conf.GetAtomPosition(atom.GetIdx())
    # Get the 3D coordinates of the neighboring atoms that are not hydrogens
    neighbor_coords = np.array(
        [
            conf.GetAtomPosition(neighbor.GetIdx())
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() != 1
        ]
    )
    # Calculate the mean of the differences between the atom coordinates and the
    # neighboring atom coordinates
    normal = np.mean(neighbor_coords - atom_coords, axis=0)
    # Normalize the normal vector to have unit length
    normal /= np.linalg.norm(normal)
    # Flip the direction of the normal vector and add it to the atom coordinates to get
    # the point on the plane
    normal *= -1
    normal += atom_coords
    return normal.tolist()


def calc_normal_aromatic(mol, ring_atoms, center):
    """
    Calculate normal vector of an aromatic ring in a molecule.

    Parameters
    ----------
    mol (rdkit.Chem.rdchem.Mol):
        An RDKit molecule object.
    ring_atoms (list of int):
        A list of atom indices that belong to the aromatic ring.
    center (numpy.ndarray):
        A NumPy array representing the 3D coordinates of the ring's center.

    Returns:
        rdkit.Geometry.Point3D:
            A Point3D object representing the point obtained by adding the normal
            vector to the center. This point can be used to define the direction of
            the normal vector in the 3D space.
    """

    # Get the 3D positions of all atoms in the molecule
    positions = mol.GetConformer().GetPositions()

    # Get the 3D positions of all atoms in the aromatic ring
    ring_positions = positions[ring_atoms]
    # center = np.mean(ring_positions, axis=0)

    # Calculate the cross product of two vectors in the ring to get the normal vector
    v1 = np.roll(ring_positions - center, -1, axis=0)
    v2 = ring_positions - center
    norm1 = np.cross(v2, v1).sum(axis=0)
    norm1 /= np.linalg.norm(norm1)
    # Create a Point3D object representing the point obtained by adding the normal
    # vector to the center
    return Point3D(*norm1 + center)
