[![CircleCI](https://circleci.com/gh/proteneer/timemachine.svg?style=svg&circle-token=d4635916d6394573ebda0aa17a63540bc8b449fc)](https://circleci.com/gh/proteneer/timemachine)

# Time Machine

This package is designed with two goals in mind:

1. Enable rapid prototyping of novel energy functions and automatic generation of gradients, hessians, and mixed partials.
2. Computes exact analytic derivatives of the trajectory with respect to model parameter, also known as backpropagation through time.

The code is implemented against the reference OpenMM Force classes, and is rigorously tested for accuracy up to machine precision.

# Example Code

The following snippet shows how one can easily compute the GBSA derivatives.

```python
import numpy as np
import tensorflow as tf
from timemachine.functionals.gbsa import GBSAOBC

masses = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
x0 = np.array([
    [ 0.0637,   0.0126,   0.2203],
    [ 1.0573,  -0.2011,   1.2864],
    [ 2.3928,   1.2209,  -0.2230],
    [-0.6891,   1.6983,   0.0780],
    [-0.6312,  -1.6261,  -0.2601]
], dtype=np.float64)

params = np.array([
    .1984, .115, .85, # H (q, r, s)
    0.0221, .19, .72  # C (q, r, s)
])

param_idxs = np.array([
    [3, 4, 5],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
])

ref_radii = ref_nrg.openmm_born_radii(x0)
nrg = GBSAOBC(params, param_idxs)

x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)
test_nrg_op = nrg.energy(x_ph)
test_nrg_grad_op = tf.gradients(test_nrg_op, x_ph)
test_nrg_hess_op = tf.hessians(test_nrg_op, x_ph)

test_nrg, test_nrg_grad, test_nrg_hess = sess.run([test_nrg_op, test_nrg_grad_op, test_nrg_hess_op], feed_dict={x_ph: x0})

# similar things can be easily done for the derivatives with respect to params
```

This is not meant to be a replacement for any production MD engine, since it's about 10-50x slower than OpenMM. For a more detailed explanation of the underlying mathematics, refer to the paper under docs for more information.

# Warning

This code is under heavy development. Expect everything to break every time you pull. An alpha release is anticipated for end of March.

# Supported Functional Forms

We currently support the following functional forms and their derivatives:

- Harmonic Bonds (bonded_force.py)
- Harmonic Angles (bonded_force.py)
- Periodic Torsions (bonded_force.py)
- Ewald Electrostatics (nonbonded_force.py)
- Ewald Leonnard Jones (nonbonded_force.py)
- Tensorfield Networks (nn_force.py)
- GBSA OBC (gbsa.py)

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
