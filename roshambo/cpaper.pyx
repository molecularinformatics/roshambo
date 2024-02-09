# cython: language_level=3

import numpy as np

from libcpp.list cimport list
from libcpp.string cimport string
from timeit import default_timer as timer

cdef extern from "GraphMol/ROMol.h" namespace "RDKit":
    cdef cppclass ROMol:
        ROMol() except +
        ROMol(const string&) except +
        void clear()

cdef extern from "../paper/paper.cu": # f"../paper/paper.cu":
    float** paper(int gpuID, list[ROMol*]& molecules)


def cpaper(gpu_id, molecules):
    num_fitmols = len(molecules) - 1
    cdef list[ROMol*] cpp_mols
    for mol in molecules:
        if mol is not None:
            cpp_mols.push_back(new ROMol(mol.to_binary()))
            # cpp_mols.push_back(new ROMol(mol.ToBinary()))

    st = timer()
    cdef float** result = paper(gpu_id, cpp_mols)
    run_time = timer() - st
    print(f"Run time: {run_time}")
    np_result = (
        np.array([[result[i][j] for j in range(16)] for i in range(num_fitmols)])
        .reshape((num_fitmols, 4, 4))
        .round(decimals=6)
    )
    return np_result
