# coding: utf-8


# Runs PAPER and calculates similarity scores.

import os
import glob
import subprocess

import numpy as np

from configparser import ConfigParser
from timeit import default_timer as timer

from scipy.spatial.transform import Rotation


class GetSimilarityScores:
    def __init__(self, ref_file, dataset_files_pattern, working_dir=None):
        self.working_dir = working_dir or os.getcwd()
        self.ref_file = f"{self.working_dir}/{ref_file}"
        self.ref_mol = Molecule()
        self.ref_mol.read_from_molfile(self.ref_file, opt=False, removeHs=False)
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")
        self.transformation_arrays = None
        self.rotation = np.array([])
        self.translation = np.array([])
        self.transformed_molecules = []

    def run_paper(self, paper_cmd=None, gpu_id=0, cleanup=True):
        run_file = f"{self.working_dir}/runfile"
        with open(run_file, "w") as f:
            for file in [self.ref_file] + self.dataset_files:
                f.write(file + "\n")

        # TODO: add mode and arguments that can be specified to paper
        # TODO: include a case where the run_file is provided as input
        if not paper_cmd:
            cfg = ConfigParser()
            cfg.read("config/config.ini")
            cmd = cfg["RunPAPER"]["paper_cmd"]
            paper_cmd = cmd.replace("$gpu_id$", str(gpu_id)).replace(
                "$run_file$", run_file
            )

        st = timer()
        return_code = subprocess.run(
            paper_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        run_time = timer() - st
        print(f"Run time: {run_time}")

        output = return_code.stdout.decode()
        output_strings = output.split("[[")
        output_strings = [i.replace("]]", "") for i in output_strings]
        output_strings = [i.replace("\n", " ") for i in output_strings]
        output_strings = [i.strip() for i in output_strings if i]

        # convert each string into a numpy array
        output_arrays = [
            np.fromstring(output_string, dtype=float, sep=" ")
            for output_string in output_strings
        ]
        self.transformation_arrays = [
            np.reshape(output_array, (4, 4)) for output_array in output_arrays
        ]

        if cleanup:
            print("Cleaning up...")
            os.remove(f"{self.working_dir}/runfile")

    def convert_transformation_arrays(self):
        # Extract rotation matrix and translation vector from transformation matrix
        for arr in self.transformation_arrays:
            r = Rotation.from_dcm(arr[:3, :3]).as_quat()
            self.rotation = np.vstack((self.rotation, r)) if self.rotation.size else r
            t = arr[:3, 3]
            self.translation = (
                np.vstack((self.translation, t)) if self.translation.size else t
            )

    def transform_molecules(self, write_to_file=False):
        # TODO: replace this since PAPER already reads molecules
        for file, rot, trans in zip(
            self.dataset_files, self.rotation, self.translation
        ):
            mol = Molecule()
            mol.read_from_molfile(file, opt=False, removeHs=False)
            xyz_trans = mol.transform_mol(rot, trans)
            mol.create_molecule(xyz_trans)
            self.transformed_molecules.append(mol)
            if write_to_file:
                mol.write_molfile(f"{file.split('.')[0]}_trans.sdf")

    def _calculate_overlap_volume(self, grid, ref_mol, fit_mol):
        gcs = grid.converted_grid
        volume = 0
        ref_mol_coords_radii = ref_mol.get_atomic_coordinates_and_radii()
        fit_mol_coords_radii = fit_mol.get_atomic_coordinates_and_radii()
        for gc in gcs:
            ref_grid = np.prod(
                [
                    1 - self.rho(ref_mol_coords_radii[i], gc)
                    for i in range(len(ref_mol_coords_radii))
                ],
                axis=0,
            )
            ref_grid = 1 - ref_grid
            fit_grid = np.prod(
                [
                    1 - self.rho(fit_mol_coords_radii[i], gc)
                    for i in range(len(fit_mol_coords_radii))
                ],
                axis=0,
            )
            fit_grid = 1 - fit_grid
            volume += ref_grid * fit_grid
        return volume * grid.res * grid.res * grid.res

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



    @staticmethod
    def rho(atom, gc):
        rt22 = 2.82842712475
        partialalpha = -2.41798793102
        alpha = partialalpha / (atom[3] ** 2)
        diff = gc - atom[:3]
        r2 = np.dot(diff, diff)
        return rt22 * np.exp(alpha * r2)
