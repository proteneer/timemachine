# Time Machine

A high-performance differentiable molecular dynamics and forcefield engine.

## Features

1. Interpolated nonbonded softcore potentials are implemented via 4D decoupling.
2. Computes derivatives of the observables with respect to forcefield parameters.

## Deficiencies

1. No constraints (HMR is needed to increase step size)
2. Explicit solvent is supported by reaction field, not PME

## Functional Forms

We currently support the following functional forms. Parameters that can be optimized are listed in parentheses.

1. HarmonicBond (force constant, bond length)
2. HarmonicAngle (force constant, ideal angle)
3. PeriodicTorsion (force constant, phase, periodicity)
4. PBC LennardJones 612 (sigma, epsilon)
5. PBC reaction field electrostatics (charge)

## Installation

### Pre-requisites

* RDKit
* OpenEye Toolkits
* OpenMM
* Cuda 12.4+
* CMake 3.24.3

### Setup using Anaconda

If using conda the following can be used to configure your environment

```shell
conda env create -f environment.yml -n timemachine
conda activate timemachine
```

### Install Time Machine

The CUDA extension module implementing custom ops is only supported on Linux, but partial functionality is still available on non-Linux OSes.

```shell
pip install -r requirements.txt
pip install .
```

## Developing Time Machine

### Installing in developer mode

```shell
pip install -r requirements.txt
pip install -e .
```

Possible variants of the last step include

```shell
pip install -e .[dev,test]                 # optionally install dev and test dependencies
CMAKE_ARGS=-DCUDA_ARCH=86 pip install -e . # override CUDA_ARCH
SKIP_CUSTOM_OPS=1 pip install -e .         # skip building CUDA extension, no effect on non-Linux OSes
```

To rebuild the extension module after making changes to the C++/CUDA code, either rerun
```shell
pip install -e .
```
or
```shell
make build  # Must have installed dev dependencies for this to work

DEBUG_BUILD=true make build  # Disable optimizations and build with debugging information
```

### Running Tests

To run tests that use `openeye`, ensure that either `OE_LICENSE` or `OE_DIR` are set appropriately.

For example, starting from a clean environment with the openeye license file in `~/.openeye`:

```shell
OE_DIR=~/.openeye pytest -xsv tests/
```

Note: we currently only support and test on python 3.12, use other versions at your own peril.

## Forcefield Gotchas

Most of the training is using the correctable charge corrections [ccc forcefield](https://github.com/proteneer/timemachine/blob/1a721dd3f05d6011cf028b0588e066682d38ba59/ff/params/smirnoff_1_1_0_ccc.py), which is SMIRNOFF 1.1.0 augmented with BCCs ported via the [recharge](https://github.com/openforcefield/openff-recharge) project. There are some additional modifications:

1. The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization.
2. The eps parameter in LJ have been replaced by an alpha such that alpha^2=eps in order to avoid negative eps values during training.
3. We use a consistent 0.5 scaling for the 1-4 terms across LJ and electrostatics.
4. The reaction field used is the real part of PME with a beta (alpha) coefficient of 2.0
5. The recharge BCC port is not yet complete, as there are some missing types that will cause very large errors (eg. P=S moieties).

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
