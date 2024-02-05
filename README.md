# <img alt="roshambo" src="docs/logo.jpg" width="500">

# Overview

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`roshambo` is a python package for robust Gaussian molecular shape comparison. It provides efficient and fast algorithms for comparing the shapes of small molecules. 
The package supports reading input files in the SDF and SMILES formats. It uses [PAPER](https://simtk.org/projects/paper/)
in the backend for overlap optimization. 

## Installation

### Prerequisites

`roshambo` requires the following packages to be installed:

- numpy
- pandas
- cython

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

