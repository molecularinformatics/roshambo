# pypaper

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`pypaper` is a python package for robust Gaussian molecular shape comparison. It provides efficient and fast algorithms for comparing the shapes of small molecules. 
The package supports input file formats including PDB, MOL, and SMILES. It uses [PAPER](https://simtk.org/projects/paper/)
 in the backend for overlap optimization. 

## Installation

### Prerequisites

`pypaper` requires the following packages to be installed:

- numpy
- pandas
- cython

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/rashatwi/pypaper.git
    ```

2. Navigate to the pypaper directory:

    ```bash
    cd pypaper
    ```

3. Install the package:

    ```bash
    python setup.py build_ext
    ```
   
## Usage

```python
import pypaper

# Call functions from the package
pypaper.some_function()

