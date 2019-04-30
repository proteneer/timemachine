[![CircleCI](https://circleci.com/gh/proteneer/timemachine.svg?style=svg&circle-token=d4635916d6394573ebda0aa17a63540bc8b449fc)](https://circleci.com/gh/proteneer/timemachine)

# Time Machine

This package is designed with two goals in mind:

1. Enable rapid prototyping of novel energy functions and automatic generation of gradients, hessians, and mixed partials.
2. Computes exact analytic derivatives of the trajectory with respect to model parameter, also known as backpropagation through time.

The code is implemented against the reference OpenMM Force classes, and is rigorously tested for accuracy up to machine precision.

# Warning

This code is under heavy development and barely working.  Expect everything to break every time you pull. An alpha release is anticipated for end of May.

# Supported Functional Forms

We currently support the following functional forms and their derivatives:

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
