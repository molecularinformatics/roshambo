import os
import json

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

from cairosvg import svg2png
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


def draw_pharm(rdkit_mol, features, filename="pharm.jpg"):
    atom_highlights = defaultdict(list)
    highlight_rads = {}
    for feature in features:
        if feature[0] in FEATURES:
            color = FEATURES[feature[0]][2]
            for atom_id in feature[1]:
                atom_highlights[atom_id].append(color)
                highlight_rads[atom_id] = 0.5

    rdDepictor.Compute2DCoords(rdkit_mol)
    rdDepictor.SetPreferCoordGen(True)
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
    # Use black for all elements
    drawer.drawOptions().updateAtomPalette(
        {k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()}
    )
    drawer.SetLineWidth(2)
    drawer.SetFontSize(6.0)
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
    with open("pharm.svg", "w") as f:
        f.write(svg)

    svg2png(bytestring=svg, write_to="image.png")

    fig, (ax, picture) = plt.subplots(
        nrows=2,
        figsize=(4, 4),
        gridspec_kw={"height_ratios": [1, 5]},
    )

    mol_image = img.imread("image.png")
    picture.imshow(mol_image)
    picture.axis("off")
    os.remove("image.png")
    os.remove("pharm.svg")

    # Data for the circles
    circle_radii = [0, 50, 100, 150, 200, 250]
    feature_values = list(FEATURES.values())
    circle_colors = [i[2] for i in feature_values]
    circle_annotations = [
        "Donor",
        "Acceptor",
        "Cation",
        "Anion",
        "Ring",
        "Hydrophobe",
    ]
    # Draw the circles and annotations
    for radius, color, annotation in zip(
        circle_radii, circle_colors, circle_annotations
    ):
        x = radius
        circle = plt.Circle((x, -5), 5, color=color)  # , alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(
            annotation,
            (x, 10),
            va="center",
            ha="center",
            fontsize=6,
            fontweight="bold",
        )

    # Set axis limits
    ax.set_xlim(-10, 270)
    ax.set_ylim(-20, 20)
    ax.axis("off")

    # Set aspect ratio to equal
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(filename, dpi=600)
