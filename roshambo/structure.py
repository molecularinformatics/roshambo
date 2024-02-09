import copy
import logging
import hashlib

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
    """A class representing a molecule.

    Attributes:
        mol (rdkit.Chem.rdchem.Mol):
            An RDKit Mol object representing the molecule.
        name (str):
            The name of the molecule, as specified by the '_Name' property in
            the mol object.

    Args:
        rdkit_mol (rdkit.Chem.rdchem.Mol):
            An RDKit Mol object representing the molecule.
    """

    def __init__(self, rdkit_mol):
        self.mol = rdkit_mol
        self.name = self.mol.GetProp("_Name")

    def get_atomic_coordinates_and_radii(self, use_carbon_radii=False):
        """Return atomic coordinates and radii for all atoms in the molecule.

        Args:
            use_carbon_radii (bool, optional):
                Whether to use the van der Waals radius of carbon for all atoms.
                Defaults to False.

        Returns:
            np.ndarray:
                A numpy array with shape (n_atoms, 4) containing the atomic
                coordinates and radii, where n_atoms is the number of atoms in
                the molecule.
        """
        # Get a list of atoms in the molecule
        atoms = self.mol.GetAtoms()
        coordinates_and_radii = []
        conf = self.mol.GetConformer()
        periodic_table = Chem.GetPeriodicTable()
        for i, j in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            if use_carbon_radii:
                # If use_carbon_radii is True, use a fixed radius of 1.7 Å for all atoms
                radius = periodic_table.GetRvdw(6)
            else:
                # If use_carbon_radii is False, use the van der Waals radius for
                # the atomic number
                radius = periodic_table.GetRvdw(j.GetAtomicNum())
            coordinates_and_radii.append((pos.x, pos.y, pos.z, radius))
        return np.array(coordinates_and_radii)

    def transform_mol(self, rot, trans):
        """
        Transforms the atomic coordinates of the molecule by rotating and
        translating the molecule.

        Args:
            rot (np.ndarray): A 4D array representing the rotation matrix.
            trans (np.ndarray): A 1D array representing the translation vector.

        Returns:
            np.ndarray: A 2D array representing the transformed atomic coordinates
            and radii of the molecule.
        """
        # Get the atomic coordinates of the molecule without the radii
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]
        # Create a rotation object from the rotation matrix
        r = R.from_quat(rot)
        # Apply the rotation to the atomic coordinates and add the translation vector
        xyz_trans = r.apply(xyz) + trans
        return xyz_trans

    def project_mol(self):
        """
        Project the atomic coordinates of a molecule onto a plane defined by the
        principal axes.

        This method computes the Singular Value Decomposition (SVD) of the atomic
        coordinates to obtain the principal axes, ensuring that the determinant of the
        transformation matrix is positive. It then projects the atomic coordinates onto
        the plane defined by these axes, and creates a new molecule object with the
        projected coordinates.

        Note that this function modifies the molecule object in place.

        Returns:
            None
        """
        # Get the atomic coordinates of the molecule without the radii
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]

        # Perform SVD to obtain the principal components of the molecule's
        # atomic coordinates
        u, s, vh = np.linalg.svd(xyz)

        # Check the determinant of the rotation matrix and correct it if necessary
        if np.linalg.det(vh) < 0:
            vh = -vh

        # Rotate the molecule to align the second and third principal components
        # with the y- and z-axes, respectively
        new_xyz = np.dot(xyz, vh.T)

        # Create a new molecule object using the new coordinates
        self.create_molecule(new_xyz)

    def center_mol(self):
        """
        Center the molecule at the origin of the coordinate system.

        The function calculates the centroid of the molecule's atomic coordinates,
        and translates the molecule so that the centroid coincides with the origin
        (0, 0, 0) of the coordinate system.

        Note that this function modifies the molecule object in place.

        Returns:
            None
        """
        # Calculate the centroid of the molecule's atomic coordinates
        centroid = np.zeros((1, 3))
        count = 0
        xyz = self.get_atomic_coordinates_and_radii()[:, :3]
        for atom in xyz:
            centroid = centroid + atom
            count += 1
        centroid = centroid / count

        # Translate the molecule by subtracting the centroid
        new_xyz = xyz - centroid

        # Create a new molecule object using the new coordinates
        self.create_molecule(new_xyz)

    def generate_conformers(
        self,
        n_confs=10,
        random_seed=999,
        method="ETKDGv3",
        ff="MMFF94s",
        add_hs=True,
        opt_confs=False,
        calc_energy=False,
        energy_iters=200,
        energy_cutoff=np.inf,
        align_confs=False,
        rms_cutoff=None,
        num_threads=1,
        rdkit_args=None,
    ):
        """Generates conformers for a molecule and optionally optimizes or aligns them.

        Args:
            n_confs (int, optional):
                The number of conformers to generate. Defaults to 10.
            random_seed (int, optional):
                The seed for the random number generator used in conformer generation.
                Defaults to 999.
            method (str, optional):
                The method for embedding conformers, one of "ETDG", "ETKDG", or
                "ETKDGv2". Defaults to "ETKDGv2".
            ff (str, optional):
                The force field to use for conformer optimization, one of "UFF",
                "MMFF94s", or "MMFF94s_noEstat". Defaults to "MMFF94s".
            add_hs (bool, optional):
                Whether to add hydrogens to the molecule before conformer generation.
                Defaults to True.
            opt_confs (bool, optional):
                Whether to optimize the generated conformers using the specified force
                field.Defaults to False.
            calc_energy (bool, optional):
                Whether to calculate the energy of the conformers without optimization.
                Defaults to False.
            energy_iters (int, optional):
                The maximum number of iterations to use in energy minimization.
                Defaults to 200.
            energy_cutoff (float, optional):
                The maximum energy difference (in kcal/mol) to keep a conformer after
                energy minimization. Energy difference is calculated between the lowest
                energy conformer and the current one. Defaults to np.inf, meaning that
                all conformers are kept.
            align_confs (bool, optional):
                Whether to align the conformers to each other using the first conformer
                as a reference.
            rms_cutoff (float, optional):
                The RMSD cutoff (in Angstroms) for clustering conformers based on
                similarity. Only the conformer corresponding to the centroid of each
                cluster is returned.
            num_threads (int, optional):
                The number of threads to use in conformer generation and optimization.
            rdkit_args (dict, optional):
                Additional arguments to pass to RDKit for conformer generation
                (rdkit.Chem.rdDistGeom.EmbedMultipleConfs)

        Raises:
            AssertionError:
                If the specified embedding method or ff is not supported.

        Returns:
            None:
                The function only modifies the molecule object in place.
        """
        assert method in [
            "ETDG",
            "ETKDG",
            "ETKDGv2",
            "ETKDGv3",
        ], f"{method} not supported for generating conformers"

        # Define parameters for embedding conformers
        args = getattr(rdDistGeom, method)()
        args.randomSeed = random_seed
        args.clearConfs = True
        args.numThreads = num_threads

        if rdkit_args:
            for k, v in rdkit_args.items():
                setattr(args, k, v)

        if add_hs:
            self.mol = Chem.AddHs(
                self.mol,
                explicitOnly=False,
                addCoords=True,
            )

        # Embed conformers
        confs = rdDistGeom.EmbedMultipleConfs(self.mol, numConfs=n_confs, params=args)
        if len(confs) == 0:
            logging.warning(f"No confs found for {self.mol_name()}")

        energies = None
        try:
            if opt_confs:
                # Optimize conformers using user supplied force field
                assert ff in [
                    "UFF",
                    "MMFF94s",
                    "MMFF94s_noEstat",
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
        except:
            logging.warning(f"Cannot optimize and/or calculate energies for {self.mol_name()}")

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
        """Returns a force field object for a specified conformer of a molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol):
                RDKit Mol object representing the molecule.
            ff (str):
                The name of the force field to use. Supported options are "UFF",
                "MMFF94s", and "MMFF94s_noEstat".
            conf_id (int, optional):
                The ID of the conformer for which to generate the force field.
                Defaults to -1, indicating the first conformer in the molecule.

        Returns:
            rdkit.ForceField.rdForceField.ForceField:
                The force field object.

        Raises:
            AssertionError:
                If an unsupported force field name is provided.
        """
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
                # Turn off the electrostatic term if using MMFF94s_noEstat
                py_mmff.SetMMFFEleTerm(False)
            return rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol, pyMMFFMolProperties=py_mmff, confId=conf_id
            )

    def process_confs(self, ff, ignore_hs):
        """Process conformers of a molecule object.

        Args:
            ff (str):
                The force field used to optimize conformers.
            ignore_hs (bool):
                Whether to remove hydrogen atoms from the generated conformers.

        Returns:
            List[Molecule]:
                A list of Molecule objects for each conformer of the input molecule.
        """
        conformers = []
        mol_copy = copy.deepcopy(self.mol)
        for i, conf in enumerate(mol_copy.GetConformers()):
            # Get conformer name by appending its index to the original mol name
            conformer_name = f"{self.mol_name()}_{i}"
            conformer_mol = Chem.Mol(mol_copy)
            conformer_mol.RemoveAllConformers()
            conformer_mol.AddConformer(Chem.Conformer(conf))
            # for prop in ["energy", "delta_energy"]:
            #     prop_name = f"rdkit_{ff}_{prop}"
            #     prop_val = conf.GetDoubleProp(prop_name)
            #     conformer_mol.SetDoubleProp(prop_name, prop_val)

            if ignore_hs:
                conformer_mol = AllChem.RemoveHs(conformer_mol)

            # For each conformer, set its name and create a new Molecule object
            conformer_mol.SetProp("_Name", conformer_name)
            conformer = Molecule(conformer_mol)

            # Center and project the conformer
            conformer.center_mol()
            conformer.project_mol()
            conformers.append(conformer)
        return conformers

    def mol_name(self):
        """Get the name of the molecule.

        Returns:
            str:
                The name of the molecule, as specified by the '_Name' property
                in the mol object.
        """
        return self.mol.GetProp("_Name")

    def to_binary(self):
        """Convert the molecule to a binary representation.

        Returns:
            bytes:
                The binary representation of the molecule.
        """
        return self.mol.ToBinary()

    def create_molecule(self, coords):
        """
        Creates a new RDKit Mol object with the provided atomic coordinates
        and updates the current Molecule object with the new Mol.

        Args:
            coords (numpy.ndarray):
                A numpy array of atomic coordinates with shape (n, 3), where n is
                the number of atoms in the molecule.

        Returns:
            None
        """
        # Get the current conformer and update its atom positions with the
        # provided coords
        conf = self.mol.GetConformer()
        for i in range(self.mol.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))

        # Create a new Mol object with the updated conformer
        new_mol = Chem.Mol(self.mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(Chem.Conformer(conf))

        self.mol = new_mol

    def get_inchikey(self):
        inchi_key = Chem.MolToInchiKey(
            self.mol, options="/FixedH /SUU /RecMet /KET /15T"
        )
        if inchi_key is None:
            return None
        return hashlib.md5(inchi_key.encode("utf-8")).hexdigest()

    def write_molfile(self, output_file):
        """Write the molecule to a file in PDB or SDF format.

        Args:
            output_file (str):
                The path to the output file. The file format is inferred from the
                file extension.

        Raises:
            ValueError:
                If the file format is not valid.
        """
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
