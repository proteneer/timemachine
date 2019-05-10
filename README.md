[![CircleCI](https://circleci.com/gh/proteneer/timemachine.svg?style=svg&circle-token=d4635916d6394573ebda0aa17a63540bc8b449fc)](https://circleci.com/gh/proteneer/timemachine)

# Time Machine

This package is designed with two goals in mind:

1. Enable rapid prototyping of novel energy functions and automatic generation of gradients, hessians, and mixed partials.
2. Computes exact analytic derivatives of the trajectory with respect to model parameter, also known as backpropagation through time.

The code is implemented against the reference OpenMM Force classes, and is rigorously tested for accuracy up to machine precision.

# Example Code:

``` python

import functools
import numpy as np
import jax

from timemachine.potentials import bonded

x0 = np.array([
    [1.0, 0.2, 3.3], # H 
    [-0.5,-1.1,-0.9], # C
    [3.4, 5.5, 0.2], # H 
], dtype=np.float64)

params = np.array([10.0, 3.0, 5.5], dtype=np.float64)

param_idxs = np.array([
    [0,1],
    [1,2],
], dtype=np.int32)

bond_idxs = np.array([
    [0,1],
    [1,2]
], dtype=np.int32)

# wrap the energy function for convenience
energy_fn = functools.partial(
    bonded.harmonic_bond,
    param_idxs=param_idxs,
    bond_idxs=bond_idxs)

# dE/dx, shape [N,3]:
dedx_fn = jax.grad(energy_fn, argnums=(0,))
dedx_fn(x0, params, box=None)

# dE/dparams, shape [3,]:
dedp_fn = jax.grad(energy_fn, argnums=(1,))
dedp_fn(x0, params, box=None)

# d^2E/dx^2, shape [N,3,N,3]:
d2edx2_fn = jax.hessian(energy_fn, argnums=(0,))
d2edx2_fn(x0, params, box=None)

# d^2E/dxde, shape [N,3,3]:
d2edxde_fn = jax.jacfwd(jax.jacrev(energy_fn, argnums=(0,)), argnums=(1,))
d2edxde_fn(x0, params, box=None)

```


# Warning

This code is under heavy development. APIs for potential energies are fairly stable now. 

# Supported Potentials

We currently support the following functional forms:

- (Periodic) Harmonic Bonds
- (Periodic) Harmonic Angles
- (Periodic) Periodic Torsions
- (Periodic) Electrostatics
- (Periodic) Leonnard Jones
- GBSA OBC

# Supported Integrators

- Langevin Dynamics
- Gradient Descent

# Requirements

Dependencies are easily pip-installable. See requirements.txt for the full list.

# Contributions

If you'd like to contribute, either email Yutong directly via githubusername@gmail.com or find him via the Open Forcefield Slack channel.

# License

Copyright 2019 Yutong Zhao

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
