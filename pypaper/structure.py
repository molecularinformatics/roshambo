import numpy as np

from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


class Molecule:
    def __init__(self):
        self.mol = None
        self.name = None

    def read_from_molfile(self, mol_file, opt=True, removeHs=False):
        self.mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs)
        # TODO: test this on xyz or pdb or other file types
        self.name = self.mol.GetProp("_Name")
        if opt:
            self.optimize_mol()

    def optimize_mol(self, addHs=True):
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(
            self.mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True
        )
        AllChem.UFFOptimizeMolecule(self.mol)

    def get_atomic_coordinates_and_radii(self):
        atoms = self.mol.GetAtoms()
        coordinates_and_radii = []
        conf = self.mol.GetConformer()
        for i, j in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            coordinates_and_radii.append(
                (pos.x, pos.y, pos.z, Chem.GetPeriodicTable().GetRvdw(j.GetAtomicNum()))
            )
        return np.array(coordinates_and_radii)

    def transform_mol(self, rot, trans):
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]
        r = R.from_quat(rot)
        xyz_trans = r.apply(xyz) + trans
        return xyz_trans

    def project_mol(self):
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]
        u, s, vh = np.linalg.svd(xyz)
        if np.linalg.det(vh) < 0:
            vh = -vh
        new_xyz = np.dot(xyz, vh.T)
        self.create_molecule(new_xyz)

    def center_mol(self):
        centroid = np.zeros((1, 3))
        count = 0
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]
        for atom in xyz:
            centroid = centroid + atom
            count += 1
        centroid = centroid / count
        new_xyz = xyz - centroid
        self.create_molecule(new_xyz)

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
