import os

from rdkit.Chem import AllChem

RDKIT_PATH = os.environ.get("RDBASE")


def calc_pharmacophore(rdkit_mol):
    # Build a feature factory using the BaseFeatures.fdef file
    feature_factory = AllChem.BuildFeatureFactory(RDKIT_PATH + "/Data/BaseFeatures.fdef")

    # Get a list of molecular features for the molecule using the feature factory
    features = feature_factory.GetFeaturesForMol(rdkit_mol)
    for feature in features:
        print(feature.GetFamily(), feature.GetAtomIds())

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
