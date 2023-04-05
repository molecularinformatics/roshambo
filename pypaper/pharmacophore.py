import os

import numpy as np
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdchem

PI = 3.14159265358

RDKIT_PATH = os.environ.get("RDBASE")

FEATURES = {
    "Donor": [1.0, True],
    "Acceptor": [1.0, True],
    "PosIonizable": [1.0, False],
    "NegIonizable": [1.0, False],
    "Aromatic": [0.7, True],
    "Hydrophobe": [1.0, False],
}


def calc_pharmacophore(rdkit_mol):
    # Build a feature factory using the BaseFeatures.fdef file
    feature_factory = AllChem.BuildFeatureFactory(
        RDKIT_PATH + "/Data/BaseFeatures.fdef"
    )

    # Get a list of molecular features for the molecule using the feature factory
    features = feature_factory.GetFeaturesForMol(rdkit_mol)
    pharmacophore = []
    for feature in features:
        fam = feature.GetFamily()
        if fam in FEATURES.keys():
            pos = feature.GetPos()
            atom_indices = feature.GetAtomIds()
            p = [
                fam,
                atom_indices,
                [pos[0], pos[1], pos[2]],
                FEATURES[fam][0],
                FEATURES[fam][1],
            ]
            if FEATURES[fam][1]:
                if fam == "Aromatic":
                    n = calc_normal_aromatic(rdkit_mol, list(atom_indices), pos)
                else:
                    n = calc_normal(rdkit_mol, rdkit_mol.GetAtoms()[atom_indices[0]])
                p.append(n)
            pharmacophore.append(p)
    return pharmacophore


def calc_single_pharm_overlap(p1, p2):
    r2 = (p1[2][0] - p2[2][0])**2 + (p1[2][1] - p2[2][1])**2 + (p1[2][2] - p2[2][2])**2
    vol = 7.999999999 * pow(PI / (p1[3] + p2[3]), 1.5)
    vol *= np.exp(-(p1[3] * p2[3]) * r2 / (p1[3] + p2[3]))
    return vol


def calc_pharm_overlap(ref_pharm, fit_pharm):
    volume = 0
    for i, p1 in enumerate(ref_pharm):
        for j, p2 in enumerate(fit_pharm):
            if p1[0] == p2[0]:
                vol = calc_single_pharm_overlap(p1, p2)
                # print(p1[0], vol)
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


def calc_multi_pharm_overlap(fit_mol, ref_pharm):
    fit_pharm = calc_pharmacophore(fit_mol.mol)
    fit_overlap = calc_pharm_overlap(fit_pharm, fit_pharm)
    ref_fit_overlap = calc_pharm_overlap(ref_pharm, fit_pharm)
    return fit_overlap, ref_fit_overlap


def color_tanimoto(ref_mol, fit_mol):
    ref_pharm = calc_pharmacophore(ref_mol)
    fit_pharm = calc_pharmacophore(fit_mol)
    ref_volume = 0
    fit_volume = 0
    overlap = 0
    for i, p1 in enumerate(ref_pharm):
        for j, p2 in enumerate(ref_pharm):
            if p1[0] == p2[0]:
                vol = calc_single_pharm_overlap(p1, p2)
                ref_volume += vol
    for i, p1 in enumerate(fit_pharm):
        for j, p2 in enumerate(fit_pharm):
            if p1[0] == p2[0]:
                vol = calc_single_pharm_overlap(p1, p2)
                fit_volume += vol
    for p1 in ref_pharm:
        for p2 in fit_pharm:
            if p1[0] == p2[0]:
                v = calc_single_pharm_overlap(p1, p2)
                # print(p1[0], p1[1], p2[1], v)
                overlap += v
    # print(overlap, ref_volume, fit_volume)
    tanimoto = overlap / (ref_volume + fit_volume - overlap)
    return tanimoto


def calc_normal(rdkit_mol, atom):
    """
    Calculate the normal vector to the plane defined by the atom and its neighboring atoms.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom

    Returns
    -------
    normal_point : list
        normal vector [x, y, z].

    Returns:
        list: normal vector [x, y, z].
    """
    conf = rdkit_mol.GetConformer()
    atom_coords = conf.GetAtomPosition(atom.GetIdx())
    neighbor_coords = np.array(
        [
            conf.GetAtomPosition(neighbor.GetIdx())
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() != 1
        ]
    )
    normal = np.mean(neighbor_coords - atom_coords, axis=0)
    normal /= np.linalg.norm(normal)
    normal *= -1
    normal += atom_coords
    return normal.tolist()


def calc_normal_aromatic(mol, ring_atoms, center):
    """
    Calculate normal vector of an aromatic ring in a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.
    ring_atoms : list of int
        A list of atom indices that belong to the aromatic ring.
    center : numpy.ndarray
        A NumPy array representing the 3D coordinates of the ring's center.

    Returns
    -------
    normal_point : rdkit.Geometry.Point3D
        A Point3D object representing the point obtained by adding the normal vector to the center.
        This point can be used to define the direction of the normal vector in the 3D space.
    """

    positions = mol.GetConformer().GetPositions()
    ring_positions = positions[ring_atoms]
    # center = np.mean(ring_positions, axis=0)
    v1 = np.roll(ring_positions - center, -1, axis=0)
    v2 = ring_positions - center
    norm1 = np.cross(v2, v1).sum(axis=0)
    norm1 /= np.linalg.norm(norm1)
    return Point3D(*norm1 + center)
