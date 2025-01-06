"""setuptools based setup module.

Adapted from https://github.com/pypa/sampleproject/blob/main/setup.py
"""

import multiprocessing
import os
import pathlib
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import versioneer


def install_custom_ops() -> bool:
    """Determine if we should install the custom ops.

    If it is not a linux machine and doesn't have at least nvcc we skip it. Can
    still use the reference platform if needed in such cases.
    """
    if os.environ.get("SKIP_CUSTOM_OPS"):
        return False
    if "linux" not in sys.platform:
        return False
    try:
        subprocess.check_call(["nvcc", "--version"])
    except FileNotFoundError:
        return False

    return True


# CMake configuration adapted from https://github.com/pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=CMakeBuild)

ext_modules = None
if install_custom_ops():
    ext_modules = [CMakeExtension("timemachine.lib.custom_ops", "timemachine/cpp")]

setup(
    name="timemachine",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="A high-performance differentiable molecular dynamics, docking and optimization engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/proteneer/timemachine",
    author="Relay Therapeutics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    keywords="molecular dynamics",
    ext_modules=ext_modules,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "jax",
        "jaxlib>0.4.1",
        "networkx",
        "numpy",
        "pymbar>=3.0.6,<4",
        "rdkit",
        "scipy",
        "matplotlib",
        "openmm",
    ],
    extras_require={
        "dev": [
            "black==23.9.1",
            "flake8==6.1.0",
            "isort==5.12.0",
            "mypy==1.5.1",
            "pre-commit==3.4.0",
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "hilbertcurve==1.0.5",
            "hypothesis[numpy]==6.54.6",
            "psutil==5.9.5",
            "py3Dmol==2.0.3",
        ],
        "viz": ["py3Dmol"],
    },
    package_data={
        "timemachine": [
            "py.typed",
        ],
        "timemachine.datasets": [
            "**/*.csv",
            "**/*.pdb",
            "**/*.sdf",
        ],
        "timemachine.testsystems": [
            "**/*.pdb",
            "**/*.sdf",
        ],
        # "timemachine.cpp": [
        #     "**/*.h",
        #     "**/*.cu",
        #     "**/*.cuh",
        #     "**/*.hpp",
        #     "**/*.cpp",
        #     "CMakeLists.txt",
        #     "generate_stubs",
        # ],
    },
    project_urls={
        "Bug Reports": "https://github.com/proteneer/timemachine/issues",
        "Source": "https://github.com/proteneer/timemachine/",
    },
)
