import os

from rdkit import Chem

try:
    from openeye import oechem
except ImportError:
    pass


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


def split_sdf_file(input_file, output_dir, max_mols_per_file=20, ignore_hydrogens=False,
                   cleanup=False):
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
    suppl = Chem.SDMolSupplier(input_file, removeHs=ignore_hydrogens)
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
