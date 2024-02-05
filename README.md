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
[documentation](https://www.rdkit.org/docs/Install.html#building-from-source) for 
installation instructions. 

> [!IMPORTANT]    
> We have tested `roshambo` with `rdkit` version 2023.03.1. We highly recommend using this version of `rdkit` to avoid any compatibility issues. 

Additionally, since `roshambo` is GPU-accelerated, you need to have CUDA installed. 

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
   
## Usage

```python
import roshambo

# Call functions from the package
roshambo.some_function()

