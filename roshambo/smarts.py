import os
import json

import numpy as np

from IPython.display import SVG
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

from roshambo.pharmacophore import FEATURES


def load_smarts_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file '{json_file}' does not exist.")
    with open(json_file, "r") as file:
        features = json.load(file)
    compiled_smarts = {k: list(map(Chem.MolFromSmarts, v)) for k, v in features.items()}
    return compiled_smarts


def compute_match_centroid(mol, matched_pattern):
    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in matched_pattern]
    center = np.mean(positions, axis=0)
    return tuple(center)


def find_matches(mol, patterns):
    matches = []
    for pattern in patterns:
        # Get all matches for that pattern
        matched = mol.GetSubstructMatches(pattern)
        for m in matched:
            # Get the centroid of each matched group
            # centroid = average_match(mol, m)
            centroid = compute_match_centroid(mol, m)
            # Add the atom indices and (x, y, z) coordinates to the list of matches
            matches.append([m, centroid])
    return matches


def calc_custom_pharm(rdkit_mol, compiled_smarts):
    matches = {}
    for key, value in compiled_smarts.items():
        matches[key] = find_matches(rdkit_mol, value)

    # Sometimes, a site can match multiple SMARTS representing the same pharmacophore,
    # so we need to keep it only once
    cleaned_matches = {}
    for key, value in matches.items():
        unique_lists = []
        for lst in value:
            if lst not in unique_lists:
                unique_lists.append(lst)
        cleaned_matches[key] = unique_lists

    pharmacophore = []
    for key, value in cleaned_matches.items():
        feature_data = FEATURES[key]
        for match in value:
            p = [key, match[0], match[1], feature_data[0], feature_data[1]]
            pharmacophore.append(p)

    return pharmacophore


def draw_pharm(rdkit_mol, feats, filename):
    colors = {
        "Donor": (1, 0.7451, 0.0431),
        "Acceptor": (0.9843, 0.3373, 0.0275),
        "PosIonizable": (1, 0, 0.4314),
        "NegIonizable": (0.5137, 0.2196, 0.9255),
        "Aromatic": (0.2275, 0.5255, 1),
        "Hydrophobe": (1, 0, 1),
    }
    atom_highlights = defaultdict(list)
    highlight_rads = {}
    for feature_type, features in feats.items():
        if feature_type in colors:
            clr = colors[feature_type]
            for aid in features:
                for atom in aid[0]:
                    atom_highlights[atom].append(clr)
                    highlight_rads[atom] = 0.5

    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDepictor.SetPreferCoordGen(True)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.drawOptions().updateAtomPalette(
        {k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()}
    )
    drawer.SetLineWidth(2)
    drawer.SetFontSize(1.0)
    drawer.drawOptions().continuousHighlight = False
    drawer.drawOptions().splitBonds = False
    drawer.drawOptions().fillHighlights = True

    for atom in rdkit_mol.GetAtoms():
        atom.SetProp("atomLabel", atom.GetSymbol())
    drawer.DrawMoleculeWithHighlights(
        rdkit_mol, "", dict(atom_highlights), {}, highlight_rads, {}
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    SVG(svg)
    with open(filename, "w") as f:
        f.write(svg)
