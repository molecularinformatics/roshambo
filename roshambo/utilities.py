import os
import copy
import time
import logging

from multiprocessing import Pool, cpu_count

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem

from roshambo.structure import Molecule

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
    Converts an OpenEye OEB file to an sdf file.

    Args:
        oeb_file (str):
            oeb file name.
        sdf_file (str):
            output SDF file name.
        working_dir (str, optional):
            Working directory where the oeb file is located.

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
    Splits an sdf file into multiple files using RDKit. Creates unique name for the
    molecules by appending an index to each molecule name that is repeated. The unique
    name is used as a file name if max_mmols_per_file is 1, otherwise the original file
    name is used with a counter appended.

    Args:
        input_file (str):
            Full path of the input sdf file.
        output_dir (str):
            Output directory where the sdf file is saved.
        max_mols_per_file (int):
            Maximum number of molecules per output file.
        ignore_hs (bool):
            Whether to ignore hydrogens. Defaults to False.
        cleanup (bool):
            Whether to delete the original sdf file. Defaults to False.

    Returns:
        List:
            List of output file paths.
    """

    # Get the base name of the input file
    name = os.path.basename(input_file).split(".sdf")[0]

    # Create a molecule supplier from the input file
    suppl = Chem.SDMolSupplier(input_file, removeHs=ignore_hs)

    # Initialize counters and output file list
    count = 0
    mol_count = 0
    writer = None
    output_files = []

    # Dictionary to store the number of molecules with each name
    mol_names = {}

    # Iterate through the molecules in the input file
    for mol in suppl:
        # Skip over any None values returned by the supplier
        if mol is None:
            continue
        # If we've hit the maximum number of molecules per output file, create a new file
        if mol_count % max_mols_per_file == 0:
            if writer is not None:
                writer.close()
            count += 1
            # If we're only writing one molecule per output file, add a number to
            # the end of the molecule name
            if max_mols_per_file == 1:
                file_name = mol.GetProp("_Name")
                if file_name in mol_names:
                    mol_names[file_name] += 1
                else:
                    mol_names[file_name] = 0
                file_name = f"{file_name}_{mol_names[file_name]}"
            # If we're writing multiple molecules per output file, append the file
            # number to the base name
            else:
                file_name = f"{name}_{count}"
            # Set the name of the molecule to the new file name
            mol.SetProp("_Name", file_name)
            # Create a new output file with the appropriate name
            file_path = f"{output_dir}/{file_name}.sdf"
            writer = Chem.SDWriter(f"{output_dir}/{file_name}.sdf")
            output_files.append(file_path)
        # Write the current molecule to the current output file
        writer.write(mol)
        mol_count += 1
    if writer is not None:
        writer.close()
    # If the cleanup flag is set, delete the original input file
    if cleanup:
        os.remove(input_file)
    return output_files


def smiles_to_rdmol(
    file_names, ignore_hs=True, name_prefix="mol", smiles_kwargs=None, embed_kwargs=None
):
    """
    Converts a list of SMILES files to a list of RDKit molecule objects.

    Args:
        file_names (list of str):
            A list of paths to SMILES files.
        ignore_hs (bool, optional):
            Whether to ignore hydrogens. Defaults to True.
        name_prefix (str, optional):
            The name to use for molecules. Molecules will be named as
            f"{name_prefix}_0", f"{name_prefix}_1", ..., f"{name_prefix}_N" if they do
            not have names in the SMILES files. Defaults to "mol".
        smiles_kwargs (dict, optional):
            Additional arguments to pass to rdkit.Chem.rdmolfiles.SmilesMolSupplier.
            Defaults to None.
        embed_kwargs (dict, optional):
            Additional arguments to pass to rdkit.Chem.AllChem.EmbedMolecule.
            Defaults to None.

    Raises:
        ValueError:
            If no valid molecules are found in the input.

    Returns:
        List[rdkit.Chem.rdchem.Mol]:
            A list of RDKit molecule objects.
    """
    rdmols = []
    used_names = {}
    if smiles_kwargs is None:
        smiles_kwargs = {}
    if embed_kwargs is None:
        embed_kwargs = {}
    for file_name in file_names:
        if not os.path.isfile(file_name):
            continue
        # Create a RDKit SmilesMolSupplier object to parse the SMILES strings
        supplier = rdmolfiles.SmilesMolSupplier(file_name, **smiles_kwargs)
        # Iterate over the molecules in the supplier
        for mol in supplier:
            if not mol:
                continue
            # Add hydrogens to the molecule
            mol = AllChem.AddHs(mol)
            # Embed the molecule in 3D
            id = AllChem.EmbedMolecule(mol, **embed_kwargs)
            if id != 0:
                logging.warning(
                    f"Embedding failed for molecule with SMILES: "
                    f"{Chem.MolToSmiles(mol)}. Skipping this molecule."
                )
                continue
            if ignore_hs:
                # Remove hydrogens from the molecule
                mol = AllChem.RemoveHs(mol)
            # Set the name of the molecule to its name found in the file or to a
            # name_prefix
            name = mol.GetProp("_Name") or name_prefix
            if name in used_names:
                used_names[name] += 1
                new_name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
                new_name = f"{name}_0"
            mol.SetProp("_Name", new_name)
            mol.SetProp("Original_Name", name)
            rdmols.append(mol)
    if not rdmols:
        raise ValueError("No valid molecules found in input.")
    return rdmols


def sdf_to_rdmol(file_names, ignore_hs=True):
    """
    Converts SDF files to RDKit molecule objects. Renames molecules to have unique names
    by appending an index to their original names parsed from the SDF files.

    Args:
        file_names (list of str):
            A list of file paths to SDF files.
        ignore_hs (bool):
            Whether to ignore hydrogens. Defaults to True.

    Returns:
        List[rdkit.Chem.rdchem.Mol]:
            A list of RDKit molecule objects.
    """

    mols = []
    used_names = {}
    for file_name in file_names:
        if not os.path.isfile(file_name):
            continue
        # Use SDMolSupplier to read in SDF file
        supplier = Chem.SDMolSupplier(file_name, removeHs=ignore_hs)
        for mol in supplier:
            if not mol:
                continue
            # Get molecule name
            name = mol.GetProp("_Name")
            # Check if the molecule name has been used before
            if name in used_names:
                used_names[name] += 1
                new_name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
                new_name = f"{name}_0"
            # Rename molecule with new name
            mol.SetProp("_Name", new_name)
            mol.SetProp("Original_Name", name)
            mols.append(mol)
    return mols


def process_molecule(rdkit_mol, ignore_hs, n_confs, keep_mol, **conf_kwargs):
    """
    Processes an RDKit molecule by:
        1. Creating a Molecule object out of the RDKit molecule
        2. Centering the Molecule to the origin
        3. Projecting the Molecule along its PCA
        4. Generating N conformers

    Args:
        rdkit_mol (rdkit.Chem.rdchem.Mol):
            An RDKit molecule object to process.
        ignore_hs (bool):
            Whether or not to ignore hydrogens in the molecule.
        n_confs (int):
            The number of conformers to generate for the molecule.
        keep_mol (bool):
            Whether or not to include the input molecule in the returned list of Molecules.
        **conf_kwargs:
            Additional keyword arguments to pass to the Molecule.generate_conformers() method.

    Returns:
        List[Molecule]:
            A list of processed Molecule objects.
    """

    # Convert RDKit molecule to Molecule object
    mol = Molecule(rdkit_mol)

    # Center and project the molecule
    mol.center_mol()
    mol.project_mol()

    # Deep copy the molecule
    new_mol = copy.deepcopy(mol)

    # Generate conformers
    if n_confs:
        new_mol.generate_conformers(n_confs, **conf_kwargs)
        conformers = new_mol.process_confs(conf_kwargs.get("ff", "UFF"), ignore_hs)

        # Return list of Molecules with input molecule
        if keep_mol:
            return [mol] + conformers
        # Return list of conformers only
        else:
            return conformers
    # Return list with input molecule only
    else:
        return [mol]


def process_molecule_with_kwargs(args, kwargs):
    """
    Calls the `process_molecule` function with the given arguments. Meant to be used
    to create a function that takes a single argument (e.g. for use with the
    multiprocessing module), where that single argument is a tuple that contains
    both the positional arguments and keyword arguments to be passed to
    process_molecule.

    Args:
        args (tuple):
            A tuple containing positional arguments for the `process_molecule` function.
        kwargs (dict):
            A dictionary containing keyword arguments for the `process_molecule` function.

    Returns:
        List[Molecule]:
            A list of processed Molecule objects returned by the
            `process_molecule` function.
    """

    return process_molecule(*args, **kwargs)


def prepare_mols(
    inputs,
    ignore_hs=True,
    n_confs=10,
    keep_mol=False,
    name_prefix="mol",
    smiles_kwargs=None,
    embed_kwargs=None,
    working_dir=None,
    **conf_kwargs,
):
    """
    Takes a list of input files (.sdf or .smi), converts them to centered and projected
    Molecule objects, and optionally generates conformers. The resulting processed
    Molecule objects are written to a SDF file called mols.sdf. The function returns
    a list of the processed Molecule objects and a list of the names of the molecules.

    Args:
        inputs (list):
            List of input molecules in the form of either a .sdf or .smiles file name.
        ignore_hs (bool):
            Whether to ignore hydrogen atoms when processing molecules.
        n_confs (int):
            Number of conformers to generate for each molecule.
        keep_mol (bool):
            Whether to keep the initial molecule in the list of processed molecules.
        name_prefix (str):
            Prefix for molecule names if not found in input.
        smiles_kwargs (dict):
            Keyword arguments to pass to rdkit.Chem.rdmolfiles.SmilesMolSupplier.
        embed_kwargs (dict):
            Keyword arguments to pass to
        **conf_kwargs (dict):
            Keyword arguments to pass to `generate_conformers` and `process_confs`
            methods of the `Molecule` class.

    Returns:
        Tuple (list, list):
            A tuple containing two lists:
            - A list of processed Molecule objects.
            - A list of molecule names.
    """

    if not working_dir:
        working_dir = os.getcwd()

    st = time.time()
    processed_mols = []
    mol_names = []
    mol_keys = []

    # Check if input file is SDF or SMILES
    is_sdf_input = any(
        os.path.isfile(input_str) and input_str.endswith(".sdf") for input_str in inputs
    )

    # Convert input file to RDKit molecules
    if is_sdf_input:
        rdmols = sdf_to_rdmol(inputs, ignore_hs=ignore_hs)
    else:
        rdmols = smiles_to_rdmol(
            inputs, ignore_hs, name_prefix, smiles_kwargs, embed_kwargs
        )

    # Process each molecule
    input_data = [(rdmol, ignore_hs, n_confs, keep_mol) for rdmol in rdmols]
    kwargs_list = [conf_kwargs] * len(input_data)
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(
            process_molecule_with_kwargs, zip(input_data, kwargs_list)
        )

    # Append processed molecules to lists
    for mols in results:
        for mol in mols:
            processed_mols.append(mol)
            mol_names.append(mol.mol.GetProp("_Name"))
            mol_keys.append(mol.get_inchikey())

    # Write processed molecules to an SDF file
    sd_writer = Chem.SDWriter(f"{working_dir}/mols.sdf")
    for mol in processed_mols:
        sd_writer.write(mol.mol)
    sd_writer.close()

    et = time.time()
    print(f"Preparing mols took: {et - st}")
    return processed_mols, mol_names, mol_keys
