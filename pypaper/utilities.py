import os
import copy
import time

from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem

from pypaper.structure import Molecule

try:
    from openeye import oechem
except ImportError:
    pass


Chem.SetDefaultPickleProperties(
    Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.ComputedProps
)
# Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.ComputedProps)


def convert_oeb_to_sdf(oeb_file, sdf_file, working_dir=None):
    """
    Convert an OpenEye OEB file to an sdf file.

    Args:
        oeb_file (str): oeb file name.
        sdf_file (str): output SDF file name.
        working_dir (str, optional): working directory where the oeb file is located.

    Returns:
        None
    """
    working_dir = working_dir or os.getcwd()
    ifs = oechem.oemolistream()
    ofs = oechem.oemolostream()

    if ifs.open(f"{working_dir}/{oeb_file}"):
        if ofs.open(f"{working_dir}/{sdf_file}"):
            for mol in ifs.GetOEGraphMols():
                oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Fatal(f"Unable to create {sdf_file}")
    else:
        oechem.OEThrow.Fatal(f"Unable to open {oeb_file}")


def split_sdf_file(
    input_file, output_dir, max_mols_per_file=20, ignore_hs=False, cleanup=False
):
    """
    Split an sdf file into multiple files using RDKit.

    Args:
        input_file (str): input sdf file name.
        output_dir (str): output directory where the sdf file is saved.
        max_mols_per_file (int): maximum number of molecules per output file.
        cleanup (bool): whether to delete the original sdf file; defaults to False.

    Returns:
        output_files (list): list of output file paths
    """
    name = os.path.basename(input_file).split(".sdf")[0]
    suppl = Chem.SDMolSupplier(input_file, removeHs=ignore_hs)
    count = 0
    mol_count = 0
    writer = None
    output_files = []
    mol_names = {}
    for mol in suppl:
        if mol is None:
            continue
        if mol_count % max_mols_per_file == 0:
            if writer is not None:
                writer.close()
            count += 1
            if max_mols_per_file == 1:
                file_name = mol.GetProp("_Name")
                if file_name in mol_names:
                    mol_names[file_name] += 1
                else:
                    mol_names[file_name] = 0
                file_name = f"{file_name}_{mol_names[file_name]}"
            else:
                file_name = f"{name}_{count}"
            mol.SetProp("_Name", file_name)
            file_path = f"{output_dir}/{file_name}.sdf"
            writer = Chem.SDWriter(f"{output_dir}/{file_name}.sdf")
            output_files.append(file_path)
        writer.write(mol)
        mol_count += 1
    if writer is not None:
        writer.close()
    if cleanup:
        os.remove(input_file)
    return output_files


def smiles_to_rdmol(
    file_names,
    ignore_hs=True,
    name_prefix="mol",
):
    rdmols = []
    used_names = {}
    for file_name in file_names:
        if not os.path.isfile(file_name):
            continue
        supplier = rdmolfiles.SmilesMolSupplier(file_name, titleLine=0)
        mols = [mol for mol in supplier if mol is not None]
        for mol in mols:
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            if ignore_hs:
                mol = AllChem.RemoveHs(mol)
            name = mol.GetProp("_Name") or name_prefix
            if name in used_names:
                used_names[name] += 1
                new_name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
                new_name = f"{name}_0"
            mol.SetProp("_Name", new_name)
            rdmols.append(mol)
    if not rdmols:
        raise ValueError("No valid molecules found in input.")
    return rdmols


def sdf_to_rdmol(file_names, ignore_hs=True):
    mols = []
    used_names = {}
    for file_name in file_names:
        if not os.path.isfile(file_name):
            continue
        suppl = Chem.SDMolSupplier(file_name, removeHs=ignore_hs)
        for mol in suppl:
            name = mol.GetProp("_Name")
            if name in used_names:
                used_names[name] += 1
                new_name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
                new_name = f"{name}_0"
            mol.SetProp("_Name", new_name)
            mols.append(mol)
    return mols


def process_molecule(rdkit_mol, opt, n_confs, keep_mol, **conf_kwargs):
    mol = Molecule(rdkit_mol, opt=opt)
    mol.center_mol()
    mol.project_mol()
    new_mol = copy.deepcopy(mol)
    if n_confs:
        new_mol.generate_conformers(n_confs, **conf_kwargs)
        conformers = new_mol.process_confs(conf_kwargs.get("ff", "UFF"))
        if keep_mol:
            return [mol] + conformers
        else:
            return conformers
    else:
        return [mol]


def prepare_mols(
    inputs,
    ignore_hs=True,
    opt=False,
    n_confs=10,
    keep_mol=False,
    **conf_kwargs,
):
    st = time.time()
    processed_mols = []
    mol_names = []

    is_sdf_input = any(
        os.path.isfile(input_str) and input_str.endswith(".sdf") for input_str in inputs
    )
    if is_sdf_input:
        rdmols = sdf_to_rdmol(inputs, ignore_hs=ignore_hs)
    else:
        rdmols = smiles_to_rdmol(inputs, ignore_hs=ignore_hs)

    with Pool() as pool:
        results = pool.starmap(
            process_molecule,
            [(rdmol, opt, n_confs, keep_mol) for rdmol in rdmols],
        )

    for mols in results:
        for mol in mols:
            processed_mols.append(mol)
            mol_names.append(mol.mol.GetProp("_Name"))

    sd_writer = Chem.SDWriter("mols.sdf")
    for mol in processed_mols:
        sd_writer.write(mol.mol)
    sd_writer.close()

    et = time.time()
    print(f"Preparing mols took: {et - st}")
    return processed_mols, mol_names
