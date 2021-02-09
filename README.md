# Time Machine

A high-performance differentiable molecular dynamics, docking and optimization engine. Computes analytical derivatives of the MD observable (eg. free energies) with respect to the forcefield parameters.

## Features

1. Nonbonded softcore potentials are implemented via 4D decoupling.
2. Computes derivatives of the observables with respect to forcefield parameters.
3. Equilibrium and non-equilibrium free energy estimates.

## Deficiencies

1. NVT only
2. No constraints (HMR is needed to increase step size)
3. Explicit solvent is supported by reaction field, not PME

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
* Cuda 11.0+

### Setup using Anaconda

If using conda the following can be used to configure your environment

```
conda create -n timemachine python=3.7
conda activate timemachine
conda install -c conda-forge -c openeye -c rdkit openmm openeye-toolkits rdkit
```

### Install Time Machine

```
pip install -r requirements.txt
cd timemachine/cpp
mkdir build
cd build
cmake -DCUDA_ARCH=sm_70 ../
make -j4 install
```

## Running Tests

Note: we currently only support and test on python 3.7, use other versions at your own peril.

```
pytest -xsv tests/
```

# Free Energy Methods

## Theory

The free energy difference of transforming B into A is defined as the log ratio of two partition functions:

<img width="187" alt="Screen Shot 2020-09-29 at 10 25 29 AM" src="https://user-images.githubusercontent.com/2280724/94571588-28c29200-023e-11eb-970a-0c03fdbcf275.png">

We estimate the free energy using [thermodynamic integration](http://www.alchemistry.org/wiki/Thermodynamic_Integration) via 4D decoupling across multiple lambda windows.

The derivative of the free energy is significantly easier and cheaper to calculate as it only requires endpoint information.

<img width="361" alt="Screen Shot 2020-09-29 at 10 25 37 AM" src="https://user-images.githubusercontent.com/2280724/94571589-28c29200-023e-11eb-9f49-bb0ee619406d.png">

When we have experimental measurements, the loss function and its derivative is therefore:

<img width="293" alt="Screen Shot 2020-09-29 at 10 25 41 AM" src="https://user-images.githubusercontent.com/2280724/94571590-28c29200-023e-11eb-8948-c8acb44eaa1a.png">

## Relative Binding Free Energy (RBFE) Training

An example source code of how one can train the RBFE via TI is available at:

- [training/rbfe_single.py](https://github.com/proteneer/timemachine/blob/master/training/rbfe_single.py)

To update the forcefield parameters, the timemachine computes derivatives of the potential with respect to redundant system parameters (C++/CUDA), which are backprop'd into unique forcefield parameters using vector jacobian products (python/jax). The parameters are fitted using gradient descent with gradient clipping whose bounds are set to physically sensible and numerically stable values for each parameter type.

## Forcefield Gotchas

Most of the training is using the correctable charge corrections [ccc forcefield](https://github.com/proteneer/timemachine/blob/master/ff/params/smirnoff_1_1_0_ccc.py), which is SMIRNOFF 1.1.0 augmented with BCCs ported via the [recharge](https://github.com/openforcefield/openff-recharge) project. There are some additional modifications:

1. The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization.
2. The eps parameter in LJ have been replaced by an alpha such that alpha^2=eps in order to avoid negative eps values during training.
3. We use a consistent 0.5 scaling for the 1-4 terms across LJ and electrostatics.
4. The reaction field used is the real part of PME with a beta (alpha) coefficient of 2.0 
5. The recharge BCC port is not yet complete, as there are some missing types that will cause very large errors (eg. P=S moeities).

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
