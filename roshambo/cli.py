import sys
import argparse

from roshambo.api import get_similarity_scores

# TODO: correct args input
# TODO: find a way to provide the kwargs
# Fix the documentation
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Get similarity scores between a reference molecule and a dataset of molecules."
    )
    parser.add_argument("ref_file", help="Name of the reference molecule file.")
    parser.add_argument(
        "dataset_files_pattern",
        help="File pattern to match the dataset molecule files.",
    )
    parser.add_argument(
        "--ignore_hs", type=bool, default=True, help="Ignore hydrogens."
    )
    parser.add_argument(
        "--n_confs", type=int, default=10, help="Number of conformers to generate."
    )
    parser.add_argument(
        "--keep_mol",
        action="store_true",
        help="Keep the original molecule in addition to the conformers.",
    )
    parser.add_argument("--working_dir", help="Working directory.")
    parser.add_argument(
        "--name_prefix",
        default="mol",
        help="Prefix to use for the molecule names if not found in the input files.",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="ID of the GPU to use for running PAPER."
    )
    parser.add_argument(
        "--volume_type",
        choices=["analytic", "gaussian"],
        default="analytic",
        help="The type of overlap volume calculation to use.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="The order of the analytic overlap volume calculation.",
    )
    parser.add_argument(
        "--proxy_cutoff",
        type=float,
        help="The distance cutoff to use for the atoms to be considered neighbors.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="The Gaussian cutoff to use in the analytic volume calculation.",
    )
    parser.add_argument(
        "--res",
        type=float,
        default=0.4,
        help="The grid resolution to use for the Gaussian volume calculation.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.4,
        help="The margin to add to the grid box size for the Gaussian volume calculation.",
    )
    parser.add_argument(
        "--use_carbon_radii",
        action="store_true",
        help="Use carbon radii for the overlap calculations.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Calculate color scores in addition to shape scores.",
    )
    parser.add_argument(
        "--fdef_path",
        help="The file path to the feature definition file to use for the pharmacophore calculation.",
    )
    parser.add_argument(
        "--sort_by",
        default="ShapeTanimoto",
        help="The score to sort the final results by.",
    )
    parser.add_argument(
        "--write_to_file",
        action="store_true",
        help="Write the transformed molecules to a sdf file.",
    )
    parser.add_argument(
        "--max_conformers",
        type=int,
        default=1,
        help="The maximum number of conformers to write for each molecule.",
    )
    parser.add_argument(
        "--filename", default="hits.sdf", help="The name of the output file to write."
    )

    args = parser.parse_args(argv)

    get_similarity_scores(
        ref_file=args.ref_file,
        dataset_files_pattern=args.dataset_files_pattern,
        ignore_hs=args.ignore_hs,
        n_confs=args.n_confs,
        keep_mol=args.keep_mol,
        working_dir=args.working_dir,
        name_prefix=args.name_prefix,
        gpu_id=args.gpu_id,
        volume_type=args.volume_type,
        n=args.n,
        proxy_cutoff=args.proxy_cutoff,
        epsilon=args.epsilon,
        res=args.res,
        margin=args.margin,
        use_carbon_radii=args.use_carbon_radii,
        color=args.color,
        fdef_path=args.fdef_path,
        sort_by=args.sort_by,
        write_to_file=args.write_to_file,
        max_conformers=args.max_conformers,
        filename=args.filename,
    )


if __name__ == "__main__":
    main()
