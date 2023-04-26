# coding: utf-8


# Runs PAPER and calculates similarity scores.

import os
import glob

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

from scipy.spatial.transform import Rotation

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
from pypaper.pharmacophore import (
    calc_pharmacophore,
    calc_pharm_overlap,
    calc_multi_pharm_overlap,
)
from pypaper.scores import scores
from pypaper.utilities import prepare_mols


class GetSimilarityScores:
    def __init__(
        self,
        ref_file,
        dataset_files_pattern,
        opt=False,
        ignore_hs=False,
        n_confs=10,
        keep_mol=False,
        working_dir=None,
        delimiter=" ",
        **conf_kwargs,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.n_confs = n_confs
        self.conf_kwargs = conf_kwargs
        self.ref_file = f"{self.working_dir}/{ref_file}"
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")

        ref_mol, _ = prepare_mols(
            [self.ref_file],
            ignore_hs=ignore_hs,
            n_confs=0,
            keep_mol=True,
            delimiter=delimiter
        )
        self.ref_mol = ref_mol[0]
        # TODO:Check if saving all molecules into numpy arrays will cause memory leaks
        self.dataset_mols, self.dataset_names = prepare_mols(
            self.dataset_files,
            ignore_hs=ignore_hs,
            n_confs=n_confs,
            keep_mol=keep_mol,
            delimiter=delimiter,
            **conf_kwargs,
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
                shape_outputs = pool.starmap(
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
                shape_outputs = pool.starmap(calc_multi_gaussian_overlap_vol, inputs)

        else:
            raise ValueError(
                "Invalid volume_type argument. Must be 'analytic' or 'gaussian'."
            )

        (
            shape_tanimoto,
            shape_fit_tversky,
            shape_ref_tversky,
            full_ref_fit_overlap,
        ) = scores(shape_outputs, ref_overlap)

        if color:
            ref_pharm = calc_pharmacophore(self.ref_mol.mol)
            ref_volume = calc_pharm_overlap(ref_pharm, ref_pharm)
            inputs = [(fit_mol, ref_pharm) for fit_mol in self.transformed_molecules]

            with Pool(processes=cpu_count()) as pool:
                outputs_pharm = pool.starmap(calc_multi_pharm_overlap, inputs)
            color_tanimoto, color_fit_tversky, color_ref_tversky, _ = scores(
                outputs_pharm, ref_volume
            )
        else:
            color_tanimoto, color_fit_tversky, color_ref_tversky = [
                np.zeros(len(self.transformed_molecules))
            ] * 3

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

        def _split_at_last_underscore(name):
            parts = name.rsplit("_", 1)
            return parts[0]

        df["Prefix"] = df["Molecule"].apply(_split_at_last_underscore)
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
                if self.n_confs:
                    ff = self.conf_kwargs.get("ff", "UFF")
                    try:
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
                        mol.mol.SetProp("PYPAPER_" + col, mol_prop)
                sd_writer.write(mol.mol)
            sd_writer.close()
        return df
