"""setuptools based setup module.

Adapted from https://github.com/pypa/sampleproject/blob/main/setup.py
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
import versioneer

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="timemachine",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "bootstrapped",
        "grpcio",
        "grpcio-tools",
        "hilbertcurve",
        "jax",
        "jaxlib",
        "networkx",
        "numpy",
        "pymbar",
        "pyyaml",
        "scipy",
    ],
    extras_require={  # Optional
        "dev": ["black==21.10b0", "pre-commit"],
        "test": ["pytest", "pytest-cov"],
    },
    # package_data={
    #     "sample": ["package_data.dat"],
    # },
    # entry_points={
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/proteneer/timemachine/issues",
        "Source": "https://github.com/proteneer/timemachine/",
    },
)
