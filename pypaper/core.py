# coding: utf-8


# Runs PAPER and calculates similarity scores.

import os
import glob

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

from scipy.spatial.transform import Rotation

from rdkit import Chem
from rdkit.Chem import AllChem

from pypaper.grid import Grid
from pypaper.cpaper import cpaper
from pypaper.grid_overlap import (
    calc_gaussian_overlap_vol,
    calc_multi_gaussian_overlap_vol,
)
from pypaper.analytic_overlap import (
    calc_analytic_overlap_vol_recursive,
    calc_multi_analytic_overlap_vol_recursive,
)
from pypaper.scores import calc_tanimoto, calc_tversky
from pypaper.pharmacophore import (
    calc_pharmacophore,
    calc_pharm_overlap,
    calc_multi_pharm_overlap,
)
from pypaper.structure import Molecule
from pypaper.utilities import prepare_mols


class GetSimilarityScores:
    def __init__(
        self,
        ref_file,
        dataset_files_pattern,
        opt=False,
        ignore_hydrogens=False,
        num_conformers=10,
        random_seed=999,
        working_dir=None,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.ref_file = f"{self.working_dir}/{ref_file}"
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")

        ref_mol, _ = prepare_mols(
            [self.ref_file],
            opt=opt,
            ignore_hydrogens=ignore_hydrogens,
            num_conformers=0,
        )
        self.ref_mol = ref_mol[0]
        # TODO:Check if saving all molecules into numpy arrays will cause memory leaks
        self.dataset_mols, self.dataset_names = prepare_mols(
            self.dataset_files,
            opt=opt,
            ignore_hydrogens=ignore_hydrogens,
            num_conformers=num_conformers,
            random_seed=random_seed,
        )

        self.transformation_arrays = None
        self.rotation = np.array([])
        self.translation = np.array([])
        self.transformed_molecules = []

    def run_paper(self, gpu_id=0):
        molecules = [self.ref_mol] + self.dataset_mols
        self.transformation_arrays = cpaper(gpu_id, molecules)

    def convert_transformation_arrays(self):
        # Extract rotation matrix and translation vector from transformation matrix
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

    def transform_molecules(self):
        for mol, rot, trans in zip(self.dataset_mols, self.rotation, self.translation):
            xyz_trans = mol.transform_mol(rot, trans)
            mol.create_molecule(xyz_trans)
            self.transformed_molecules.append(mol)

    def calculate_tanimoto(
        self,
        volume_type="analytic",
        n=2,
        proxy_cutoff=None,
        epsilon=0.1,
        res=0.4,
        margin=0.4,
        use_carbon_radii=True,
        color=False,
        sort_by="ShapeTanimoto",
        write_to_file=False,
        max_conformers=1,
        filename="hits.sdf",
    ):
        if volume_type == "analytic":
            ref_overlap = calc_analytic_overlap_vol_recursive(
                self.ref_mol,
                self.ref_mol,
                n=n,
                proxy_cutoff=proxy_cutoff,
                epsilon=epsilon,
                use_carbon_radii=use_carbon_radii,
            )
            inputs = [
                (self.ref_mol, fit_mol, n, proxy_cutoff, epsilon, use_carbon_radii)
                for fit_mol in self.transformed_molecules
            ]

            with Pool(processes=cpu_count()) as pool:
                outputs = pool.starmap(
                    calc_multi_analytic_overlap_vol_recursive, inputs
                )

        elif volume_type == "gaussian":
            ref_grid = Grid(
                self.ref_mol, res=res, margin=margin, use_carbon_radii=use_carbon_radii
            )
            ref_grid.create_grid()
            ref_overlap = calc_gaussian_overlap_vol(
                self.ref_mol, self.ref_mol, ref_grid, use_carbon_radii
            )
            inputs = [
                (fit_mol, res, margin, ref_grid, self.ref_mol, use_carbon_radii)
                for fit_mol in self.transformed_molecules
            ]

            with Pool(processes=cpu_count()) as pool:
                outputs = pool.starmap(calc_multi_gaussian_overlap_vol, inputs)

        else:
            raise ValueError(
                "Invalid volume_type argument. Must be 'analytic' or 'gaussian'."
            )

        if color:
            ref_pharm = calc_pharmacophore(self.ref_mol.mol)
            ref_volume = calc_pharm_overlap(ref_pharm, ref_pharm)
            inputs = [(fit_mol, ref_pharm) for fit_mol in self.transformed_molecules]

            with Pool(processes=cpu_count()) as pool:
                outputs_pharm = pool.starmap(calc_multi_pharm_overlap, inputs)

            # TODO: move this to a function since it is used again
            outputs_pharm = np.array(outputs_pharm)
            full_fit_overlap = outputs_pharm[:, 0]
            full_ref_fit_overlap = outputs_pharm[:, 1]
            full_ref_overlap = np.ones_like(full_fit_overlap) * ref_volume
            color_tanimoto = calc_tanimoto(
                full_ref_overlap, full_fit_overlap, full_ref_fit_overlap
            )
            color_fit_tversky = calc_tversky(
                full_ref_overlap,
                full_fit_overlap,
                full_ref_fit_overlap,
                alpha=0.05,
                beta=0.95,
            )
            color_ref_tversky = calc_tversky(
                full_ref_overlap,
                full_fit_overlap,
                full_ref_fit_overlap,
                alpha=0.95,
                beta=0.05,
            )
        else:
            color_tanimoto, color_fit_tversky, color_ref_tversky = [
                np.zeros(len(self.transformed_molecules))
            ] * 3

        outputs = np.array(outputs)
        full_fit_overlap = outputs[:, 0]
        full_ref_fit_overlap = outputs[:, 1]
        full_ref_overlap = np.ones_like(full_fit_overlap) * ref_overlap

        shape_tanimoto = calc_tanimoto(
            full_ref_overlap, full_fit_overlap, full_ref_fit_overlap
        )
        shape_fit_tversky = calc_tversky(
            full_ref_overlap,
            full_fit_overlap,
            full_ref_fit_overlap,
            alpha=0.05,
            beta=0.95,
        )
        shape_ref_tversky = calc_tversky(
            full_ref_overlap,
            full_fit_overlap,
            full_ref_fit_overlap,
            alpha=0.95,
            beta=0.05,
        )

        df_data = {
            "Molecule": self.dataset_names,
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

        df = pd.DataFrame(df_data)
        df["Prefix"] = df["Molecule"].str.split("_").str[0]
        df = df.sort_values(by=["Prefix", sort_by], ascending=[True, False])
        idx = (
            df.groupby("Prefix")
            .apply(lambda x: x.nlargest(max_conformers, sort_by))
            .index.levels[1]
        )
        df = df.loc[idx].sort_values(by=sort_by, ascending=False).round(3)
        del df["Prefix"]
        df.to_csv(f"{self.working_dir}/pypaper.csv", index=False, sep="\t")

        mol_dict = {
            _mol.mol.GetProp("_Name"): _mol for _mol in self.transformed_molecules
        }
        reordered_mol_list = [
            mol_dict[name] for name in df["Molecule"] if name in mol_dict
        ]

        if write_to_file:
            sd_writer = AllChem.SDWriter(f"{self.working_dir}/{filename}")
            df_columns = df.columns
            df = df.set_index("Molecule")
            for mol in [self.ref_mol] + reordered_mol_list:
                if mol != self.ref_mol:
                    mol_props = df.loc[mol.mol.GetProp("_Name"), :]
                    for col in df_columns[1:]:
                        mol_prop = str(mol_props[col])
                        mol.mol.SetProp("PYPAPER_" + col, mol_prop)
                sd_writer.write(mol.mol)
        return df
