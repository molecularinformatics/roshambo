import copy
import logging
import numpy as np

from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdForceFieldHelpers
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdDistGeom


class Molecule:
    def __init__(self, rdkit_mol):
        self.mol = rdkit_mol
        self.name = self.mol.GetProp("_Name")

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

    def generate_conformers(
        self,
        n_confs=10,
        random_seed=999,
        method="ETKDGv2",
        ff="UFF",
        opt_confs=False,
        calc_energy=False,
        energy_iters=200,
        energy_cutoff=np.inf,
        align_confs=False,
        rms_cutoff=None,
        num_threads=1,
        rdkit_args=None,
    ):

        assert method in [
            "ETDG",
            "ETKDG",
            "ETKDGv2",
            #"ETKDGv3",
        ], f"{method} not supported for generating conformers"

        # Define parameters for embedding conformers
        args = getattr(rdDistGeom, method)()
        args.randomSeed = random_seed
        args.clearConfs = True
        args.numThreads = num_threads

        if rdkit_args:
            for k, v in rdkit_args.items():
                setattr(args, k, v)

        # Embed conformers
        confs = rdDistGeom.EmbedMultipleConfs(self.mol, numConfs=n_confs, params=args)
        if len(confs) == 0:
            logging.warning(f"No confs found for {self.mol_name()}")

        energies = None
        if opt_confs:
            # Optimize conformers using user supplied force field
            assert ff in [
                "UFF",
                "MMFF94s",
            ], f"{ff} not supported for optimizing conformers"
            results = None
            if ff == "UFF":
                results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                    self.mol, numThreads=num_threads, maxIters=energy_iters
                )
            elif ff == "MMFF94s":
                results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                    self.mol,
                    numThreads=num_threads,
                    maxIters=energy_iters,
                    mmffVariant="MMFF94s",
                )
            energies = [energy for not_converged, energy in results]

        elif calc_energy:
            # Compute energy of conformers without opt
            energies = []
            for conf in self.mol.GetConformers():
                force_field = self._get_conf_ff(self.mol, ff, conf_id=conf.GetId())
                energy = force_field.CalcEnergy()
                energies.append(energy)

        if energies:
            min_energy = np.min(energies)
            # Add energy to conf properties
            for energy, conf in zip(energies, self.mol.GetConformers()):
                conf.SetDoubleProp(f"rdkit_{ff}_energy", energy)
                conf.SetDoubleProp(f"rdkit_{ff}_delta_energy", energy - min_energy)

            # Sort confs by energy and remove ones above energy_cutoff
            mol_copy = copy.deepcopy(self.mol)
            sorted_confs = [
                conf
                for energy, conf in sorted(
                    zip(energies, mol_copy.GetConformers()), key=lambda x: x[0]
                )
                if energy - min_energy <= energy_cutoff
            ]
            self.mol.RemoveAllConformers()
            [self.mol.AddConformer(conf, assignId=True) for conf in sorted_confs]

        if align_confs:
            # Align confs to each other using the first conf as the ref
            rdMolAlign.AlignMolConformers(self.mol)

        if rms_cutoff:
            # Calculate RMS matrix of the confs of the molecule
            dist_matrix = AllChem.GetConformerRMSMatrix(
                self.mol, prealigned=align_confs
            )
            # Cluster the data points and return a tuple of tuples:
            # ((cluster1_elem1, cluster_1_elem2, ...),
            # (cluster2_elem1, cluster2_elem2, ...)
            conf_clusters = Butina.ClusterData(
                dist_matrix,
                nPts=self.mol.GetNumConformers(),
                distThresh=rms_cutoff,
                isDistData=True,
                reordering=False,
            )
            # Get centroid of each cluster (first element in each tuple)
            centroid_list = [indices[0] for indices in conf_clusters]
            mol_copy = copy.deepcopy(self.mol)
            # Keep only the centroid conformers
            confs = [mol_copy.GetConformers()[i] for i in centroid_list]
            self.mol.RemoveAllConformers()
            [self.mol.AddConformer(conf, assignId=True) for conf in confs]

    @staticmethod
    def _get_conf_ff(mol, ff, conf_id=-1):
        assert ff in [
            "UFF",
            "MMFF94s",
            "MMFF94s_noEstat",
        ], f"{ff} not supported for optimizing conformers"
        if ff == "UFF":
            return rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf_id)
        else:
            py_mmff = rdForceFieldHelpers.MMFFGetMoleculeProperties(
                mol, mmffVariant="MMFF94s"
            )
            if ff == "MMFF94s_noEstat":
                py_mmff.SetMMFFEleTerm(False)
            return rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol, pyMMFFMolProperties=py_mmff, confId=conf_id
            )

    def process_confs(self, ff):
        conformers = []
        mol_copy = copy.deepcopy(self.mol)
        for i, conf in enumerate(mol_copy.GetConformers()):
            conformer_name = f"{self.mol_name()}_{i}"
            conformer_mol = Chem.Mol(mol_copy)
            conformer_mol.RemoveAllConformers()
            conformer_mol.AddConformer(Chem.Conformer(conf))
            for prop in ["energy", "delta_energy"]:
                prop_name = f"rdkit_{ff}_{prop}"
                prop_val = conf.GetDoubleProp(prop_name)
                conformer_mol.SetDoubleProp(prop_name, prop_val)
            conformer_mol.SetProp("_Name", conformer_name)
            conformer = Molecule(conformer_mol)
            conformer.center_mol()
            conformer.project_mol()
            conformers.append(conformer)
        return conformers

    def mol_name(self):
        return self.mol.GetProp("_Name")

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
