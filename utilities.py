import os

from rdkit import Chem
from openeye import oechem


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


def split_sdf_file(input_file, output_dir, max_mols_per_file=20):
    """
    Split an sdf file into multiple files.

    Args:
        input_file (str): input sdf file name.
        output_dir (str): output directory where the sdf file is located.
        max_mols_per_file (int): maximum number of molecules per output file.

    Returns:
        None
    """
    suppl = Chem.ForwardSDMolSupplier(input_file, removeHs=False)
    count = 0
    file_count = 0
    w = Chem.SDWriter(f"{output_dir}/part_{file_count}.sdf")

    for mol in suppl:
        if mol is None:
            continue
        w.write(mol)
        count += 1
        if count % max_mols_per_file == 0:
            w.flush()
            w.close()
            file_count += 1
            w = Chem.SDWriter(f'{output_dir}/part_{file_count}.sdf')
    w.flush()
    w.close()


