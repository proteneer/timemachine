# Time Machine

A high-performance differentiable molecular dynamics and optimization engine. Computes analytical derivatives of the MD observable (eg. free energies) with respect to the forcefield parameters.

## Features

1. Supports 4D decoupling of nonbonded terms.
2. Analytical first order derivatives of the potential with respect to the coordinates, forcefield parameters, and lambda
3. vjp support for backpropping system derivatives into forcefield derivatives.

## Deficiencies

1. NVT only
2. No constraints (HMR is needed to increase step size)
3. Explicit solvent is supported by reaction field, not PME
4. Slower than most MD packages

## Functional Forms

We currently support the following functional forms. Parameters that can be optimized are listed in parentheses.

1. HarmonicBond (force constant, bond length)
2. HarmonicAngle (force constant, ideal angle)
3. PeriodicTorsion (force constant, phase, periodicity)
4. PBC LennardJones 612 (sigma, epsilon)
5. PBC reaction field electrostatics (charge)

## Running Tests

When running tests, it's important we set the fixed point to a sufficient level of precision via BUILD_TEST=ON. This is done with CMake flags using instructions below. Important: When running actual simulations, it's important to set BUILD_TEST=OFF else overflows will occur.

```
pip install -r requirements.txt
cd timemachine/cpp
mkdir build
cd build
cmake -DBUILD_TEST=ON -DCUDA_ARCH=sm_70 ../
make -j4 install
cd ../../../
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

## Hydration Free Energy Implementation

Relevant code:

- [training/hydration_cfg.ini](https://github.com/proteneer/timemachine/blob/master/training/hydration_cfg.ini) configuration file
- [training/hydration_fe.py](https://github.com/proteneer/timemachine/blob/master/training/hydration_fe.py) main training client
- [training/worker.py](https://github.com/proteneer/timemachine/blob/master/training/worker.py) worker file

To estimate the free energy on the FreeSolv dataset, the following 30-window lambda schedule is used:

```
lambda_schedule=0.0,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.3,0.33,0.36,0.39,0.42,0.45,0.48,0.51,0.54,0.57,0.6,0.69,0.78,0.87,0.96,1.05,1.14,1.23,1.32,1.41,1.5,1.944,2.389,2.833,3.278,3.722,4.167,4.611,5.056,5.5
```

To update the forcefield parameters, the timemachine computes derivatives of the potential with respect to system parameters (C++/CUDA), which are backprop'd into forcefield parameters using vector jacobian products (python). The parameters are fitted using stochastic gradient descent with gradient clipping whose bounds are set to physically sensible and numerically stable values for each parameter type. A 50/50 train/test index split is done on the dataset to monitor overfitting. 

The main training client connects to individual workers via a client-server model through gRPC. The IP address and port of each worker is specified in the config file. The workers must be launched before the main hydration_fe.py script can be run. On each run, the worker periodically estimates the <dU/dl> and <dU/dparams>. Note that the current example only re-trains the LJ and BCC types.

## Forcefield Gotchas

Most of the training is using the correctable charge corrections [ccc forcefield](https://github.com/proteneer/timemachine/blob/master/ff/params/smirnoff_1_1_0_ccc.py), which is SMIRNOFF 1.1.0 augmented with BCCs ported via the [recharge](https://github.com/openforcefield/openff-recharge) project. There are some additional modifications:

1. The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization.
2. The eps parameter in LJ have been replaced by an alpha such that alpha^2=eps in order to avoid negative eps values during training.
3. We use a consistent 0.5 scaling for the 1-4 terms across LJ and electrostatics.
4. The reaction field used is the real part of PME with a beta coefficient of 2.0 
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
