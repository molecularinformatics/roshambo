# <img alt="roshambo" src="docs/logo.jpg" width="500">

# Overview

`roshambo` is a python package for robust Gaussian molecular shape comparison. It 
provides efficient and fast algorithms for comparing the shapes of small molecules. 
The package supports reading input files in the SDF and SMILES formats. 
It uses [PAPER](https://simtk.org/projects/paper/) in the backend for overlap 
optimization. 

## Installation

We recommend that you create a new conda environment before installing `roshambo`. 

```bash
conda create -n roshambo python=3.9
conda activate roshambo
```

### Prerequisites

`roshambo` requires a compiled version of `rdkit`. You need to compile `rdkit` with 
the same version of python that you are using to install `roshambo`. You also need to 
compile `rdkit` with the `INCHI` option enabled. Please refer to the `rdkit` 
[documentation](https://www.rdkit.org/docs/Install.html#building-from-source) and [blog](https://greglandrum.github.io/rdkit-blog/posts/2023-03-17-setting-up-a-cxx-dev-env2.html) for 
installation instructions. 

> [!IMPORTANT]    
> We have tested `roshambo` with `rdkit` version 2023.03.1. We highly recommend using this version of `rdkit` to avoid any compatibility issues. 

Additionally, since `roshambo` is GPU-accelerated, you need to have `CUDA` installed. 

After you have installed `rdkit` and `CUDA`, you need to set the following environment 
variables:

```bash
export RDBASE=/path/to/your/rdkit/installation
export RDKIT_LIB_DIR=$RDBASE/lib
export RDKIT_INCLUDE_DIR=$RDBASE/Code
export RDKIT_DATA_DIR=$RDBASE/Data
export PYTHONPATH=$PYTHONPATH:$RDBASE

export CUDA_HOME=/path/to/your/cuda/installation
```

Below is a qick ROSHAMBO installation guide. Referencing this [blog post](https://iwatobipen.wordpress.com/2024/08/08/new-cheminformatics-package-for-molecular-alignment-and-3d-similarity-scoring-cheminformatics-rdkit-memo/) from N. B. Taka might also be helpful for new users.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/rashatwi/roshambo.git
    ```

2. Navigate to the roshambo directory:

    ```bash
    cd roshambo
    ```

3. Install the package:

    ```bash
    pip3 install .
    ```
    Depending on your cluster/machine settings, you might need install in editable mode:
   
    ```bash
    pip3 install -e .
    ```
   
## Usage

```python
from roshambo.api import get_similarity_scores

get_similarity_scores(
    ref_file="query.sdf",
    dataset_files_pattern="dataset.sdf",
    ignore_hs=True,
    n_confs=0,
    use_carbon_radii=True,
    color=True,
    sort_by="ComboTanimoto",
    write_to_file=True,
    gpu_id=0,
    working_dir="data/basic_run",
)
```
The above code will run a similarity calculation between the reference molecule in 
`query.sdf` and all the molecules in the `dataset.sdf` file. Hydrogen atoms will be 
ignored when aligning the molecules and carbon radii will be used. No conformers will 
be generated for the dataset molecules. Both shape and color similarity scores will be 
calculated and the results will be written to a file. The scores will be sorted by the 
`ComboTanimoto` score and saved in the directory `data/basic_run`. 

You can also run the above example from the command line:

```bash
roshambo --n_confs 0 --ignore_hs --color --sort_by ComboTanimoto --write_to_file --working_dir data/basic_run --gpu_id 0 query.sdf dataset.sdf
```
