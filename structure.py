import os

import numpy as np

from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


class Molecule:
    def __init__(self):
        self.mol = None

    def read_from_molfile(self, mol_file, opt=True, removeHs=False):
        self.mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs)
        if opt:
            self.optimize_mol()

    def optimize_mol(self, addHs=True):
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(self.mol)

    def get_xyz_from_mol(self):
        xyz = np.zeros((self.mol.GetNumAtoms(), 3))
        conf = self.mol.GetConformer()
        for i in range(conf.GetNumAtoms()):
            position = conf.GetAtomPosition(i)
            xyz[i, 0] = position.x
            xyz[i, 1] = position.y
            xyz[i, 2] = position.z
        return xyz

    def transform_mol(self, rot, trans):
        xyz = self.get_xyz_from_mol()
        r = R.from_quat(rot)
        xyz_trans = r.apply(xyz) + trans
        return xyz_trans

    def create_molecule(self, coords):
        conf = self.mol.GetConformer()
        for i in range(self.mol.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        new_mol = Chem.Mol(self.mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(Chem.Conformer(conf))
        self.mol = new_mol

    def write_molfile(self, output_file):
        if ".pdbqt" in output_file:
            writer = Chem.PDBWriter(output_file)
            writer.write(self.mol)
            writer.close()
        elif ".pdb" in output_file:
            writer = Chem.PDBWriter(output_file)
            writer.write(self.mol)
            writer.close()
        elif ".sdf" in output_file:
            writer = Chem.SDWriter(output_file)
            writer.write(self.mol)
            writer.close()
        else:
            raise ValueError("Invalid file format")

