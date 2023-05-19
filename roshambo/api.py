from roshambo.core import GetSimilarityScores


def get_similarity_scores(
    ref_file,
    dataset_files_pattern,
    ignore_hs=False,
    n_confs=10,
    keep_mol=False,
    working_dir=None,
    name_prefix="mol",
    smiles_kwargs=None,
    embed_kwargs=None,
    gpu_id=0,
    volume_type="analytic",
    n=2,
    proxy_cutoff=None,
    epsilon=0.1,
    res=0.4,
    margin=0.4,
    use_carbon_radii=True,
    color=False,
    fdef_path=None,
    sort_by="ShapeTanimoto",
    write_to_file=False,
    max_conformers=1,
    filename="hits.sdf",
    **conf_kwargs,
):
    """
    Calculates the similarity scores between a reference molecule and a set of
    molecules in a dataset. Runs PAPER (https://doi.org/10.1002/jcc.21307)
    in the background to optimize the shape overlap. Uses the shape-optimized
    structures to compute the color (chemistry) similarity scores.

    Args:
        ref_file (str):
            Name of the reference molecule file.
        dataset_files_pattern (str):
            File pattern to match the dataset molecule files.
        ignore_hs (bool, optional):
            Whether to ignore hydrogens. Defaults to False.
        n_confs (int, optional):
            Number of conformers to generate. Defaults to 10.
        keep_mol (bool, optional):
            Whether to keep the original molecule in addition to the conformers.
            Defaults to False.
        working_dir (str, optional):
            Working directory. All output files will be written to this directory.
            Defaults to the current directory.
        name_prefix (str, optional):
            Prefix to use for the molecule names if not found in the input files.
            Defaults to "mol".
        smiles_kwargs (dict, optional):
            Additional keyword arguments to pass to the `smiles_to_rdmol` function.
        embed_kwargs (dict, optional):
            Additional keyword arguments to pass to the `smiles_to_rdmol` function.
        gpu_id (int, optional):
            ID of the GPU to use for running PAPER. Defaults to 0.
        volume_type (str, optional):
            The type of overlap volume calculation to use. Options are 'analytic'
            or 'gaussian'. Defaults to 'analytic'.
        n (int, optional):
            The order of the analytic overlap volume calculation. Defaults to 2.
        proxy_cutoff (float, optional):
            The distance cutoff to use for the atoms to be considered neighbors
            and for which overlap volume will be calculated in the analytic
            volume calculation. If not provided, will compute neighboring atoms based
            on this codition: |R_i - R_j| <= sigma_i + sigma_j + eps. Defaults to None.
        epsilon (float, optional):
            The Gaussian cutoff to use in this condition:
            |R_i - R_j| <= sigma_i + sigma_j + eps in the analytic volume
            calculation. R corresponds to the atomic coordinates, sigma is the
            radius. The larger the epsilon, the greater the number of neighbors
            each atom will have, so that in the limit of large epsilon, each atom
            will have all the remaining atoms as neighbors. Defaults to 0.1.
        res (float, optional):
            The grid resolution to use for the Gaussian volume calculation.
            Defaults to 0.4.
        margin (float, optional):
            The margin to add to the grid box size for the Gaussian volume
            calculation. Defaults to 0.4.
        use_carbon_radii (bool, optional):
            Whether to use carbon radii for the overlap calculations.
            Defaults to True.
        color (bool, optional):
            Whether to calculate color scores in addition to shape scores.
            Defaults to False.
        fdef_path (str, optional):
            The file path to the feature definition file to use for the pharmacophore
            calculation. Uses BaseFeatures.fdef if not provided. Defaults to None.
        sort_by (str, optional):
            The score to sort the final results by. Defaults to 'ShapeTanimoto'.
        write_to_file (bool, optional):
            Whether to write the transformed molecules to a sdf file.
            Defaults to False.
        max_conformers (int, optional):
            The maximum number of conformers to write for each molecule.
            Defaults to 1, meaning that only the best conformer structure will
            be written.
        filename (str, optional):
            The name of the output file to write. Defaults to 'hits.sdf'.
        **conf_kwargs (dict, optional):
            Additional keyword arguments to pass to the `generate_conformers` function.
    """
    sm = GetSimilarityScores(
        ref_file=ref_file,
        dataset_files_pattern=dataset_files_pattern,
        ignore_hs=ignore_hs,
        n_confs=n_confs,
        keep_mol=keep_mol,
        working_dir=working_dir,
        name_prefix=name_prefix,
        smiles_kwargs=smiles_kwargs,
        embed_kwargs=embed_kwargs,
        **conf_kwargs,
    )
    sm.run_paper(gpu_id=gpu_id)
    sm.convert_transformation_arrays()
    sm.transform_molecules()
    sm.calculate_scores(
        volume_type=volume_type,
        n=n,
        proxy_cutoff=proxy_cutoff,
        epsilon=epsilon,
        res=res,
        margin=margin,
        use_carbon_radii=use_carbon_radii,
        color=color,
        fdef_path=fdef_path,
        sort_by=sort_by,
        write_to_file=write_to_file,
        max_conformers=max_conformers,
        filename=filename,
    )
