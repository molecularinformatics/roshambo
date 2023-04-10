import numpy as np

from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


class Molecule:
    def __init__(self, rdkit_mol, opt=False):
        self.mol = rdkit_mol
        # TODO: test this on xyz or pdb or other file types
        self.name = self.mol.GetProp("_Name")
        if opt:
            self.optimize_mol()

    def optimize_mol(self, add_hydrogens=False):
        if add_hydrogens:
            self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(
            self.mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True
        )
        AllChem.UFFOptimizeMolecule(self.mol)

    def get_atomic_coordinates_and_radii(self, use_carbon_radii=False):
        atoms = self.mol.GetAtoms()
        coordinates_and_radii = []
        conf = self.mol.GetConformer()
        periodic_table = Chem.GetPeriodicTable()
        for i, j in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            if use_carbon_radii:
                radius = periodic_table.GetRvdw(6)
            else:
                radius = periodic_table.GetRvdw(j.GetAtomicNum())
            coordinates_and_radii.append((pos.x, pos.y, pos.z, radius))
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

    def generate_conformers(self, n_confs=10, random_seed=999):
        AllChem.EmbedMultipleConfs(
            self.mol,
            numConfs=n_confs,
            randomSeed=random_seed,
            clearConfs=True,
            numThreads=0,
        )
        # AllChem.UFFOptimizeMoleculeConfs(self.mol, numThreads=0)

    def to_binary(self):
        return self.mol.ToBinary()

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
