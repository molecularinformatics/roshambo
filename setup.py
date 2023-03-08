import os

import setuptools
from Cython.Distutils import build_ext
from distutils.extension import Extension

module_dir = os.path.dirname(os.path.abspath(__file__))

RDBASE = os.environ.get("RDBASE")
RDKIT_INCLUDE_DIR = os.path.join(RDBASE, "Code")
RDKIT_LIB_DIR = os.path.join(RDBASE, "lib")
MyRDKit_FIND_COMPONENTS = ["GraphMol", "SmilesParse", "FileParsers", "Depictor"]
RDKIT_LIBRARIES = []
for component in MyRDKit_FIND_COMPONENTS:
    print(f"Looking for RDKit component {component}")
    library_path = os.path.join(RDBASE, "lib", f"libRDKit{component}.so")
    if not os.path.isfile(library_path):
        raise Exception(f"Didn't find RDKit {component} library.")
    RDKIT_LIBRARIES.append(library_path)

PAPER_DIR = "/UserUCDD/ratwi/pypaper/paper/"
OBPATH = "/UserUCDD/ratwi/openbabel/include/openbabel-2.0"
CCFLAGS = [
    "-O2",
    "-I" + OBPATH,
    "-I" + RDKIT_INCLUDE_DIR,
    "-DORIG_GLOBAL",
    "-DFAST_OVERLAP",
    "-DNO_DIV_ADDRESS",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
]
PTXFLAGS = ["-Xcompiler", "-O2", "-arch", "sm_50", "-Xptxas", "-v"]
LDFLAGS = ["-L" + RDKIT_LIB_DIR, "-L" + "/UserUCDD/ratwi/openbabel/lib"]


def locate_cuda():
    home = "/usr/local/cuda-11/"
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
        elif src == module_dir + "/pypaper/cpaper.cpp":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = [
                "-I" + RDKIT_INCLUDE_DIR,
                "-I" + OBPATH,
                "-x",
                "cu",
                "-std=c++11",
                "-arch=sm_70",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
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
    name="cpaper",
    sources=[
        module_dir + "/pypaper/cpaper.pyx",
        PAPER_DIR + "deviceAnalyticVolume.cu",
        PAPER_DIR + "hostAnalyticVolume.cu",
        PAPER_DIR + "deviceOverlay.cu",
        PAPER_DIR + "transformTools.cu",
        PAPER_DIR + "inputFileReader.cpp",
        PAPER_DIR + "inputPreprocessor.cpp",
        PAPER_DIR + "inputModule.cu",
    ],
    include_dirs=[CUDA["include"], PAPER_DIR, RDKIT_INCLUDE_DIR],
    library_dirs=[CUDA["lib64"], PAPER_DIR, RDKIT_LIB_DIR],
    libraries=["cudart", "openbabel", "RDKitGraphMol"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"], PAPER_DIR],
    extra_compile_args={
        "gcc": CCFLAGS + ["-DGPP"] + ["-std=c++11"],
        "nvcc": CCFLAGS
        + ["-std=c++11"]
        + PTXFLAGS
        + ["--ptxas-options=-v", "-c", "--compiler-options", "-fPIC"],
    },
    extra_link_args=LDFLAGS,
    extra_objects=RDKIT_LIBRARIES,
)

setuptools.setup(
    name="pypaper",
    version="0.0.1",
    author="Rasha Atwi",
    author_email="rasha.atwi@biogen.edu",
    description="pypaper contains is a python package for robust Gaussian molecular "
    "shape comparison",
    long_description=open(os.path.join(module_dir, "README.md")).read(),
    url="https://github.com/rashatwi/pypaper",
    install_requires=["numpy", "pandas", "cython"],
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
    ext_modules=[ext],
    cmdclass={"build_ext": CustomBuildExt},
)
