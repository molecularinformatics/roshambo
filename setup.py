import os

import setuptools

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# module_dir = os.path.dirname(os.path.abspath(__file__))

env_vars = ["RDBASE", "RDKIT_INCLUDE_DIR", "RDKIT_LIB_DIR", "RDKIT_DATA_DIR"]
for var in env_vars:
    if var not in os.environ:
        raise Exception(f"{var} environment variable not set.")

RDBASE = os.environ.get("RDBASE")
RDKIT_INCLUDE_DIR = os.environ.get("RDKIT_INCLUDE_DIR")
RDKIT_LIB_DIR = os.environ.get("RDKIT_LIB_DIR")

MyRDKit_FIND_COMPONENTS = ["GraphMol", "SmilesParse", "FileParsers", "Depictor"]
RDKIT_LIBRARIES = []
for component in MyRDKit_FIND_COMPONENTS:
    print(f"Looking for RDKit component {component}")
    library_path = os.path.join(RDKIT_LIB_DIR, f"libRDKit{component}.so")
    if not os.path.isfile(library_path):
        raise Exception(f"Didn't find RDKit {component} library.")
    RDKIT_LIBRARIES.append(library_path)

PAPER_DIR = "paper"  # os.path.join(module_dir, "paper")
CCFLAGS = [
    "-O2",
    f"-I{RDKIT_INCLUDE_DIR}",
    "-DORIG_GLOBAL",
    "-DFAST_OVERLAP",
    "-DNO_DIV_ADDRESS",
    # "-D_GLIBCXX_USE_CXX11_ABI=0",  # comment
]
PTXFLAGS = ["-Xcompiler", "-O2", "-arch", "sm_50", "-Xptxas", "-v"]
LDFLAGS = [
    f"-L{RDKIT_LIB_DIR}",
]


def locate_cuda():
    home = os.environ.get("CUDA_HOME")
    if not home:
        raise Exception("CUDA_HOME environment variable not set.")
    nvcc = os.path.join(home, "bin", "nvcc")
    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": os.path.join(home, "include"),
        "lib64": os.path.join(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )
    return cudaconfig


def customize_compiler_for_nvcc(self):
    self.src_extensions.append(".cu")
    default_compiler_so = self.compiler_so
    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcc"]
        elif src == os.path.join(
            "roshambo", "cpaper.cpp"
        ):  # os.path.join(module_dir, "roshambo", "cpaper.cpp"):
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = [
                f"-I{RDKIT_INCLUDE_DIR}",
                "-x",
                "cu",
                "-std=c++17",
                "-arch=sm_70",
                # "-D_GLIBCXX_USE_CXX11_ABI=0",  # comment
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "-fPIC",
            ]
        else:
            postargs = extra_postargs["gcc"]

        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


CUDA = locate_cuda()

ext = Extension(
    name="roshambo.cpaper",
    sources=[
        os.path.join(
            "roshambo", "cpaper.pyx"
        ),  # os.path.join(module_dir, "roshambo", "cpaper.pyx"),
        os.path.join(PAPER_DIR, "deviceAnalyticVolume.cu"),
        os.path.join(PAPER_DIR, "hostAnalyticVolume.cu"),
        os.path.join(PAPER_DIR, "deviceOverlay.cu"),
        os.path.join(PAPER_DIR, "transformTools.cu"),
        os.path.join(PAPER_DIR, "inputFileReader.cpp"),
        os.path.join(PAPER_DIR, "inputPreprocessor.cpp"),
        os.path.join(PAPER_DIR, "inputModule.cu"),
    ],
    include_dirs=[CUDA["include"], PAPER_DIR, RDKIT_INCLUDE_DIR],
    library_dirs=[CUDA["lib64"], PAPER_DIR, RDKIT_LIB_DIR],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"], PAPER_DIR],
    extra_compile_args={
        "gcc": CCFLAGS + ["-DGPP"] + ["-std=c++17"],
        "nvcc": CCFLAGS
        + ["-std=c++17"]
        + PTXFLAGS
        + ["--ptxas-options=-v", "-c", "--compiler-options", "-fPIC"],
    },
    extra_link_args=LDFLAGS,
    extra_objects=RDKIT_LIBRARIES,
)

# Load requirements.txt
with open(
    "requirements.txt"
) as f:  # open(os.path.join(module_dir, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="roshambo",
    version="0.0.1",
    author="Rasha Atwi",
    author_email="rasha.atwi@biogen.edu",
    description="roshambo is a python package for robust Gaussian molecular "
    "shape comparison",
    long_description=open(
        "README.md"
    ).read(),  # open(os.path.join(module_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rashatwi/roshambo",
    install_requires=requirements,
    build_backend="setuptools.build_meta",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "roshambo = roshambo.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={"roshambo": ["roshambo/*.cpython*.so"]},
    ext_modules=[ext],
    cmdclass={"build_ext": CustomBuildExt},
)
