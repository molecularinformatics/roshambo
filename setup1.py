import os

import setuptools

module_dir = os.path.dirname(os.path.abspath(__file__))

setuptools.setup(
    name="pypaper",
    version="0.0.1",
    author="Rasha Atwi",
    author_email="rasha.atwi@biogen.edu",
    description="pypaper contains is a python package for robust Gaussian molecular "
    "shape comparison",
    long_description=open(os.path.join(module_dir, "README.md")).read(),
    url="https://github.com/rashatwi/pypaper",
    install_requires=[
        "numpy",
        "pandas",
        "cython"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.5",
    package_data={},
    # ext_modules=cythonize(ext),
)
