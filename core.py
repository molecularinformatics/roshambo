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
        self.dataset_files = glob.glob(f"{self.working_dir}/{dataset_files_pattern}")
        self.transformation_arrays = None
        self.rotation = np.array([])
        self.translation  = np.array([])

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
            paper_cmd = cmd.replace("$gpu_id$", str(gpu_id)).replace("$run_file$", run_file)

        st = timer()
        return_code = subprocess.run(paper_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        run_time = timer() - st
        print(f"Run time: {run_time}")

        output = return_code.stdout.decode()
        output_strings = output.split("[[")
        output_strings = [i.replace("]]", "") for i in output_strings]
        output_strings = [i.replace("\n", " ") for i in output_strings]
        output_strings = [i.strip() for i in output_strings if i]

        # convert each string into a numpy array
        output_arrays = [np.fromstring(output_string, dtype=float, sep=' ') for output_string in output_strings]
        self.transformation_arrays = [np.reshape(output_array, (4, 4)) for output_array in output_arrays]

        if cleanup:
            print("Cleaning up...")
            os.remove(f"{self.working_dir}/runfile")

    def convert_transformation_arrays(self):
        # Extract rotation matrix and translation vector from transformation matrix
        for arr in self.transformation_arrays:
            r = Rotation.from_dcm(arr[:3, :3]).as_quat()
            self.rotation = np.vstack((self.rotation, r)) if self.rotation.size else r
            t = arr[:3, 3]
            self.translation = np.vstack((self.translation, t)) if self.translation.size else t

    def read_molecules(self):
        # TODO: replace this since PAPER already reads molecules
        pass

    def transform_molecules(self):
        pass

    def calculate_similarity_scores(self):
        pass






