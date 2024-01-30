# coding: utf-8

# Runs PAPER and calculates similarity scores.

import os
import glob
import time

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

from scipy.spatial.transform import Rotation

from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo.grid import Grid
from roshambo.cpaper import cpaper
from roshambo.grid_overlap import (
    calc_gaussian_overlap_vol,
    calc_multi_gaussian_overlap_vol,
)
from roshambo.analytic_overlap import (
    calc_analytic_overlap_vol_recursive,
    calc_multi_analytic_overlap_vol_recursive,
)
from roshambo.pharmacophore import (
    calc_pharm,
    calc_pharm_overlap,
    calc_multi_pharm_overlap,
)
from roshambo.scores import scores
from roshambo.utilities import prepare_mols


class GetSimilarityScores:
    """
    Calculates the similarity scores between a reference molecule and a set of
    molecules in a dataset. Runs PAPER (https://doi.org/10.1002/jcc.21307)
    in the background to optimize the shape overlap.

    Args:
        ref_file (str):
            Name of the reference molecule file.
        dataset_files_pattern (str):
            File pattern to match the dataset molecule files.
        ignore_hs (bool, optional):
            Whether to ignore hydrogens. Defaults to False.
        n_confs (int, optional):
            Number of conformers to generate. Defaults to 10.
        keep_mol (bool, optional):
            Whether to keep the original molecule in addition to the conformers.
            Defaults to False.
        working_dir (str, optional):
            Working directory. All output files will be written to this directory.
            Defaults to the current directory.
        name_prefix (str, optional):
            Prefix to use for the molecule names if not found in the input files.
            Defaults to "mol".
        smiles_kwargs (dict, optional):
            Additional keyword arguments to pass to the `smiles_to_rdmol` function.
        embed_kwargs (dict, optional):
            Additional keyword arguments to pass to the `smiles_to_rdmol` function.
        **conf_kwargs (dict, optional):
            Additional keyword arguments to pass to the `generate_conformers` function.

    Attributes:
        working_dir (str):
            Working directory.
        n_confs (int):
            Number of conformers to generate.
        conf_kwargs (dict):
            Dictionary containing additional keyword arguments to pass to the
            `generate_conformers` function.
        ref_file (str):
            Path to the reference molecule file.
        dataset_files (list of str):
            List of dataset molecule files matching the given pattern.
        ref_mol (Molecule):
            Reference Molecule object.
        dataset_mols (list of Molecule):
            List of Molecule objects in the dataset.
        dataset_names (list of str):
            List of molecule names in the dataset.
        transformation_arrays (list of numpy.ndarray):
            List of transformation arrays to align the dataset molecules to the
            reference molecule.
        rotation (numpy.ndarray):
            3x3 rotation matrix used to align the dataset molecules to the
            reference molecule.
        translation (numpy.ndarray):
            3D translation vector used to align the dataset molecules to the
            reference molecule.
        transformed_molecules (list of Molecule):
            List of Molecule objects in the dataset aligned to the reference molecule.
    """

    def __init__(
        self,
        ref_file,
        dataset_files_pattern,
        ignore_hs=False,
        n_confs=10,
        keep_mol=False,
        working_dir=None,
        name_prefix="mol",
        smiles_kwargs=None,
        embed_kwargs=None,
        **conf_kwargs,
    ):
        # TODO: replace conf_kwargs with conf_dict
        # TODO: add draw_pharm as parameter to core fun
        self.working_dir = working_dir or os.getcwd()
        self.n_confs = n_confs
        self.conf_kwargs = conf_kwargs
        self.ref_file = f"{self.working_dir}/{ref_file}"
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")

        ref_mol, _, _ = prepare_mols(
            [self.ref_file],
            ignore_hs=ignore_hs,
            n_confs=0,
            keep_mol=True,
            name_prefix="ref",
            smiles_kwargs=smiles_kwargs,
            embed_kwargs=embed_kwargs,
            working_dir=self.working_dir,
        )
        self.ref_mol = ref_mol[0]
        # TODO:Check if saving all molecules into numpy arrays will cause memory leaks
        self.dataset_mols, self.dataset_names, self.dataset_keys = prepare_mols(
            self.dataset_files,
            ignore_hs=ignore_hs,
            n_confs=n_confs,
            keep_mol=keep_mol,
            name_prefix=name_prefix,
            smiles_kwargs=smiles_kwargs,
            embed_kwargs=embed_kwargs,
            working_dir=self.working_dir,
            **conf_kwargs,
        )

        self.transformation_arrays = None
        self.rotation = np.array([])
        self.translation = np.array([])
        self.transformed_molecules = []

        print(f"Total number of molecules for transformation: {len(self.dataset_mols)}")

    def run_paper(self, gpu_id=0):
        """
        Runs the PAPER package to compute the transformation arrays that result in the
        highest shape overlap between the dataset molecules and the reference molecule.
        Uses the BFGS optimizer.

        Args:
            gpu_id (int, optional):
                ID of the GPU to use. Defaults to 0.

        Returns:
            None.
        """
        st = time.time()
        molecules = [self.ref_mol] + self.dataset_mols
        self.transformation_arrays = cpaper(gpu_id, molecules)
        et = time.time()
        print(f"Running paper took: {et -st}")

    def convert_transformation_arrays(self):
        """
        Converts transformation matrices obtained from PAPER into rotation matrices and
        translation vectors.

        Returns:
            None
        """
        # Extract rotation matrix and translation vector from transformation matrix
        st = time.time()
        for arr in self.transformation_arrays:
            r = Rotation.from_matrix(arr[:3, :3]).as_quat()
            self.rotation = (
                np.vstack((self.rotation, r)) if self.rotation.size else r.reshape(1, 4)
            )
            t = arr[:3, 3]
            self.translation = (
                np.vstack((self.translation, t))
                if self.translation.size
                else t.reshape(1, 3)
            )
        et = time.time()
        print(f"Converting transformation arrays took: {et - st}")

    def transform_molecules(self):
        """
        Transforms the molecules in the dataset using the rotation and translation
        matrices. The transformed molecules are stored in the `transformed_molecules`
        attribute.

        Returns:
            None
        """
        st = time.time()
        for mol, rot, trans in zip(self.dataset_mols, self.rotation, self.translation):
            # Transform the molecule using the rotation and translation matrices
            xyz_trans = mol.transform_mol(rot, trans)
            # Create a new Molecule object with the transformed coordinates
            mol.create_molecule(xyz_trans)
            # Append the transformed Molecule object to the list of transformed molecules
            self.transformed_molecules.append(mol)
        et = time.time()
        print(f"Transforming molecules took: {et - st}")

    def calculate_scores(
        self,
        volume_type="analytic",
        n=2,
        proxy_cutoff=None,
        epsilon=0.1,
        res=0.4,
        margin=0.4,
        use_carbon_radii=True,
        color=False,
        fdef_path=None,
        sort_by="ShapeTanimoto",
        write_to_file=False,
        max_conformers=1,
        filename="hits.sdf",
    ):
        """
        Calculates shape and/or color similarity scores for transformed molecules.

        Args:
            volume_type (str, optional):
                The type of overlap volume calculation to use. Options are 'analytic'
                or 'gaussian'. Defaults to 'analytic'.
            n (int, optional):
                The order of the analytic overlap volume calculation. Defaults to 2.
            proxy_cutoff (float, optional):
                The distance cutoff to use for the atoms to be considered neighbors
                and for which overlap volume will be calculated in the analytic
                volume calculation. If not provided, will compute neighboring atoms based
                on this condition: |R_i - R_j| <= sigma_i + sigma_j + eps.
                Defaults to None.
            epsilon (float, optional):
                The Gaussian cutoff to use in this condition:
                |R_i - R_j| <= sigma_i + sigma_j + eps in the analytic volume
                calculation. R corresponds to the atomic coordinates, sigma is the
                radius, and epsilon is an arbitrary parameter called the Gaussian
                cutoff. The larger the epsilon, the greater the number of neighbors
                each atom will have, so that in the limit of large epsilon, each atom
                will have all the remaining atoms as neighbors. Defaults to 0.1.
            res (float, optional):
                The grid resolution to use for the Gaussian volume calculation.
                Defaults to 0.4.
            margin (float, optional):
                The margin to add to the grid box size for the Gaussian volume
                calculation. Defaults to 0.4.
            use_carbon_radii (bool, optional):
                Whether to use carbon radii for the overlap calculations.
                Defaults to True.
            color (bool, optional):
                Whether to calculate color scores in addition to shape scores.
                Defaults to False.
            fdef_path (str, optional):
                The file path to the feature definition file to use for the pharmacophore
                calculation. Uses BaseFeatures.fdef if not provided. Defaults to None.
            sort_by (str, optional):
                The column to sort the final results by. Defaults to 'ShapeTanimoto'.
            write_to_file (bool, optional):
                Whether to write the transformed molecules to a sdf file.
                Defaults to False.
            max_conformers (int, optional):
                The maximum number of conformers to write for each molecule.
                Defaults to 1, meaning that only the best conformer structure will
                be written.
            filename (str, optional):
                The name of the output file to write. Defaults to 'hits.sdf'.

        Raises:
            ValueError:
                If the `volume_type` argument is not 'analytic' or 'gaussian'.

        Returns:
            pandas.DataFrame:
                A dataframe containing the similarity scores and overlap volumes for each
                transformed molecule.
        """

        st = time.time()
        if volume_type == "analytic":
            # Calculate reference self shape overlap volume using analytic method
            ref_overlap = calc_analytic_overlap_vol_recursive(
                self.ref_mol,
                self.ref_mol,
                n=n,
                proxy_cutoff=proxy_cutoff,
                epsilon=epsilon,
                use_carbon_radii=use_carbon_radii,
            )
            # Calculate fit self shape overlap volume and shape overlap volume between
            # reference and fit molecule using analytic method
            inputs = [
                (self.ref_mol, fit_mol, n, proxy_cutoff, epsilon, use_carbon_radii)
                for fit_mol in self.transformed_molecules
            ]

            with Pool(processes=cpu_count()) as pool:
                shape_outputs = pool.starmap(
                    calc_multi_analytic_overlap_vol_recursive, inputs
                )
        elif volume_type == "gaussian":
            # Create grid and calculate reference self shape overlap volume using
            # gaussian method
            ref_grid = Grid(
                self.ref_mol, res=res, margin=margin, use_carbon_radii=use_carbon_radii
            )
            ref_grid.create_grid()
            ref_overlap = calc_gaussian_overlap_vol(
                self.ref_mol, self.ref_mol, ref_grid, use_carbon_radii
            )

            # Calculate fit self shape overlap volume and shape overlap volume between
            # reference and fit molecule using gaussian method
            inputs = [
                (fit_mol, res, margin, ref_grid, self.ref_mol, use_carbon_radii)
                for fit_mol in self.transformed_molecules
            ]

            with Pool(processes=cpu_count()) as pool:
                shape_outputs = pool.starmap(calc_multi_gaussian_overlap_vol, inputs)

        else:
            raise ValueError(
                "Invalid volume_type argument. Must be 'analytic' or 'gaussian'."
            )

        # Calculate shape tanimoto and tversky scores
        (
            shape_tanimoto,
            shape_fit_tversky,
            shape_ref_tversky,
            full_ref_fit_overlap,
        ) = scores(shape_outputs, ref_overlap)
        et = time.time()
        print(f"Calculating shape scores took: {et - st}")

        # Calculate color scores if color flag is set to True
        if color:
            st = time.time()
            # Calculate the pharmacophore features of the reference molecule and its
            # self-overlap volume for color scoring
            ref_pharm = calc_pharm(self.ref_mol.mol, fdef_path)
            ref_volume = calc_pharm_overlap(ref_pharm, ref_pharm)
            # Calculate fit self color overlap volume and color overlap volume between
            # reference and fit molecule
            inputs = [
                (fit_mol, ref_pharm, fdef_path)
                for fit_mol in self.transformed_molecules
            ]

            with Pool(processes=cpu_count()) as pool:
                outputs_pharm = pool.starmap(calc_multi_pharm_overlap, inputs)

            # Calculate color tanimoto and tversky scores
            color_tanimoto, color_fit_tversky, color_ref_tversky, _ = scores(
                outputs_pharm, ref_volume
            )
            et = time.time()
            print(f"Calculating color scores took: {et - st}")
        else:
            color_tanimoto, color_fit_tversky, color_ref_tversky = [
                np.zeros(len(self.transformed_molecules))
            ] * 3

        st = time.time()
        # Prepare the final df data
        df_data = {
            "Molecule": self.dataset_names,
            "OriginalName": [
                mol.mol.GetProp("Original_Name") for mol in self.dataset_mols
            ],
            "Hash": self.dataset_keys,
            "ComboTanimoto": shape_tanimoto + color_tanimoto,
            "ShapeTanimoto": shape_tanimoto,
            "ColorTanimoto": color_tanimoto,
            "FitTverskyCombo": shape_fit_tversky + color_fit_tversky,
            "FitTversky": shape_fit_tversky,
            "FitColorTversky": color_fit_tversky,
            "RefTverskyCombo": shape_ref_tversky + color_ref_tversky,
            "RefTversky": shape_ref_tversky,
            "RefColorTversky": color_ref_tversky,
            "Overlap": full_ref_fit_overlap,
        }
        # Create the final df with the scores and molecule names
        df = pd.DataFrame(df_data)
        df = df.sort_values(by=sort_by, ascending=False)
        idx = (
            df.groupby("Hash")
            .apply(lambda x: x.nlargest(max_conformers, sort_by))
            .index.levels[1]
        )
        df = df.loc[idx].sort_values(by=sort_by, ascending=False).round(3)
        del df["Hash"]
        df.to_csv(f"{self.working_dir}/roshambo.csv", index=False, sep="\t")
        del df["OriginalName"]
        et = time.time()
        print(f"Creating dataframe took: {et - st}")

        st = time.time()
        mol_dict = {
            _mol.mol.GetProp("_Name"): _mol for _mol in self.transformed_molecules
        }
        # Creates a list of molecule objects sorted according to the order of
        # 'Molecule' column in df dataframe
        reordered_mol_list = [
            mol_dict[name] for name in df["Molecule"] if name in mol_dict
        ]

        # If write_to_file is True, writes the molecule data to an SDF file with the
        # specified filename
        if write_to_file:
            sd_writer = AllChem.SDWriter(f"{self.working_dir}/{filename}")
            df_columns = df.columns
            df = df.set_index("Molecule")
            # Loops over each molecule object in the reordered_mol_list and writes it
            # to the SDF file
            for mol in [self.ref_mol] + reordered_mol_list:
                if self.n_confs:
                    ff = self.conf_kwargs.get("ff", "UFF")
                    try:
                        # Sets the 'rdkit_ff_energy' and 'rdkit_ff_delta_energy'
                        # properties of the molecule, if conformers were generated
                        mol.mol.SetProp(
                            f"rdkit_{ff}_energy", mol.mol.GetProp(f"rdkit_{ff}_energy")
                        )
                        mol.mol.SetProp(
                            f"rdkit_{ff}_delta_energy",
                            mol.mol.GetProp(f"rdkit_{ff}_delta_energy"),
                        )
                    except Exception:
                        pass
                if mol != self.ref_mol:
                    mol_props = df.loc[mol.mol.GetProp("_Name"), :]
                    for col in df_columns[1:]:
                        mol_prop = str(mol_props[col])
                        mol.mol.SetProp("ROSHAMBO_" + col, mol_prop)
                mol_with_hs = Chem.AddHs(mol.mol, addCoords=True)
                sd_writer.write(mol_with_hs)
            sd_writer.close()
        et = time.time()
        print(f"Writing molecule file took: {et - st}")
        return df
