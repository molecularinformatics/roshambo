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


