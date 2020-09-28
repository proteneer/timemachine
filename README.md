# Time Machine

A high-performance differentiable molecular dynamics and optimization engine. Computes analytical derivatives of the MD observable (eg. free energies) with respect to the forcefield parameters.

## Features

1. Optimized for performance on modern NVIDIA GPUs.
2. Analytical first order derivatives of the potential with respect to the coordinates, forcefield parameters, and alchemical lambda via 4D decoupling
3. Integrated forcefield support, allowing one to compute derivatives of equilibrium thermodynamic observables.

## Functional Forms

We currently support the following functional forms. Parameters that can be optimized are listed in parentheses.

1. HarmonicBond (force constant, bond length)
2. HarmonicAngle (force constant, ideal angle)
3. PeriodicTorsion (force constant, phase, periodicity)
4. PBC LennardJones 612 (sigma, epsilon)
5. PBC reaction field electrostatics (charge)

## Running Tests

Note: when running tests, it's important we set the fixed point to a sufficient level of precision via BUILD_TEST=ON. This is done with CMake flags using instructions below. When running actual simulations, it's important to set BUILD_TEST=OFF since it will cause overflows.

```
pip install -r requirements.txt
cd timemachine/cpp
mkdir build
cd build
cmake -DBUILD_TEST=ON -DCUDA_ARCH=sm_70 ../
make -j4 install
cd ../../
pytest -xsv tests/
```

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
