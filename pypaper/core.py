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

from pypaper.grid import Grid
from pypaper.cpaper import cpaper
from pypaper.grid_overlap import (
    calc_gaussian_overlap_vol,
    calc_multi_gaussian_overlap_vol,
)
from pypaper.analytic_overlap import (
    calc_analytic_overlap_vol,
    calc_multi_analytic_overlap_vol,
    calc_analytic_overlap_vol_recursive,
    calc_multi_analytic_overlap_vol_recursive,
)
from pypaper.structure import Molecule
from pypaper.utilities import split_sdf_file


class GetSimilarityScores:
    def __init__(
        self,
        ref_file,
        dataset_files_pattern,
        split_dataset_files=False,
        opt=False,
        ignore_hydrogens=False,
        working_dir=None,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.ref_file = f"{self.working_dir}/{ref_file}"
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")

        # TODO: this only works if the input is an sdf file, what about other mol format?
        if split_dataset_files:
            assert len(self.dataset_files) == 1
            self.dataset_files = split_sdf_file(
                self.dataset_files[0],
                output_dir=self.working_dir,
                max_mols_per_file=1,
                cleanup=True,
            )

        # TODO:Check if saving all molecules into numpy arrays will cause memory leaks
        self.ref_mol = self._process_molecule(
            self.ref_file, opt=opt, ignore_hydrogens=ignore_hydrogens
        )
        self.dataset_mols = [
            self._process_molecule(file, opt=opt, ignore_hydrogens=ignore_hydrogens)
            for file in self.dataset_files
        ]

        self.transformation_arrays = None
        self.rotation = np.array([])
        self.translation = np.array([])
        self.transformed_molecules = []

    @staticmethod
    def _process_molecule(file, opt=False, ignore_hydrogens=False):
        rdkit_mol = Chem.MolFromMolFile(file, removeHs=ignore_hydrogens)
        mol = Molecule(rdkit_mol, opt=opt)
        mol.center_mol()
        mol.project_mol()
        mol.write_molfile(file)
        return mol

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

    def transform_molecules(self, write_to_file=False):
        # TODO: replace this since PAPER already reads molecules
        for mol, rot, trans in zip(self.dataset_mols, self.rotation, self.translation):
            xyz_trans = mol.transform_mol(rot, trans)
            mol.create_molecule(xyz_trans)
            self.transformed_molecules.append(mol)
            if write_to_file:
                mol.write_molfile(f"{self.working_dir}/{mol.name}.sdf")

    def calculate_volume(self, grid, mol):
        gcs = grid.converted_grid
        volume = 0
        mol_coords_radii = mol.get_atomic_coordinates_and_radii()
        for gc in gcs:
            mol_grid = np.prod(
                [
                    1 - self.rho(mol_coords_radii[i], gc)
                    for i in range(len(mol_coords_radii))
                ],
                axis=0,
            )
            volume += 1 - mol_grid
        return volume * grid.res**3

    def calculate_tanimoto(
        self, volume_type="analytic", res=0.4, margin=0.4, save_to_file=False
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
            # ref_overlap = calc_analytic_overlap_vol(self.ref_mol, self.ref_mol)
            inputs = [(self.ref_mol, fit_mol) for fit_mol in self.transformed_molecules]

            with Pool(processes=cpu_count()) as pool:
                # outputs = pool.starmap(calc_multi_analytic_overlap_vol, inputs)
                outputs = pool.starmap(
                    calc_multi_analytic_overlap_vol_recursive, inputs
                )

        elif volume_type == "gaussian":
            st = time.time()
            ref_grid = Grid(self.ref_mol, res=res, margin=margin)
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
            et = time.time()
            print(f"Gaussian volume calculation took: {et - st}")

        else:
            raise ValueError(
                "Invalid volume_type argument. Must be 'analytic' or 'gaussian'."
            )

        outputs = np.array(outputs)
        full_fit_overlap = outputs[:, 0]
        full_ref_fit_overlap = outputs[:, 1]
        full_ref_overlap = np.ones_like(full_fit_overlap) * ref_overlap
        full_tanimoto = full_ref_fit_overlap / (
            full_ref_overlap + full_fit_overlap - full_ref_fit_overlap
        )

        df = pd.DataFrame(
            {
                "Molecule": [
                    os.path.basename(path).split(".")[0] for path in self.dataset_files
                ],
                "Overlap": full_ref_fit_overlap,
                "Tanimoto": full_tanimoto,
            }
        )
        df.sort_values(by="Tanimoto", ascending=False, inplace=True)
        if save_to_file:
            df.to_csv(f"{self.working_dir}/tanimoto.csv", index=False)
        return df
