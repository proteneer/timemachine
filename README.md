# Time Machine

A high-performance differentiable molecular dynamics and optimization engine.

## Features

1. Optimized for performance on modern NVIDIA GPUs.
2. Analytical first order derivatives of the potential with respect to the coordinates and the forcefield parameters.
3. Analytical second order hessian vector products and mixed vector products of the above at a 2.5x cost.
4. Implements adjoint equations of motion via rematerialization, enabling one to differentiate objective functions with respect to an arbitrary number of forcefield parameters in a *single* backwards pass.
5. Supports 3 and 4 dimensional coordinates, enabling geometric decoupling for alchemical methods and docking-like protocols.
6. Mixed forcefield support, with AMBER for protein and OFF for small molecules.

## Functional Forms

We currently support the following functional forms. Parameters that can be optimized are listed in parentheses.

1. HarmonicBond (force constant, bond length)
2. HarmonicAngle (force constant, ideal angle)
3. PeriodicTorsion (force constant, phase, periodicity)
4. LennardJones 612 (sigma, epsilon)
5. Non-periodic electrostatics (charge)
6. GBSA (charge, atomic radii, atomic scale factors)

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
