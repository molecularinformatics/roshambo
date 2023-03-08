import numpy as np

from rdkit import Chem
from libcpp.list cimport list
from libcpp.string cimport string
from timeit import default_timer as timer

cdef extern from "GraphMol/ROMol.h" namespace "RDKit":
    cdef cppclass ROMol:
        ROMol() except +
        ROMol(const string&) except +
        void clear()

cdef extern from "/UserUCDD/ratwi/pypaper/paper/paper.cu":
    float** paper(int gpuID, list[ROMol*]& molecules)

def cpaper(gpu_id, molfiles):
    molfiles_b = [arg.encode("utf-8") for arg in molfiles]

    cdef list[ROMol*] cpp_mols
    for file in molfiles:
        supplier = Chem.SDMolSupplier(file)
        for mol in supplier:
            if mol is not None:
                cpp_mols.push_back(new ROMol(mol.ToBinary()))

    num_fitmols = len(molfiles) - 1

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
