[![CircleCI](https://circleci.com/gh/proteneer/timemachine.svg?style=svg&circle-token=d4635916d6394573ebda0aa17a63540bc8b449fc)](https://circleci.com/gh/proteneer/timemachine)

# The Time Machine

Computes analytic derivatives of the trajectory with respect to model parameters. Also allows for rapid prototype of novel functional forms.

This is a prototype written by Yutong during his winter break period of funemployment. For a more detailed explanation, refer to the paper under docs for more information.

# Supported Functional Forms

We currently support only a subset of most forcefields. These are implemented by their energy functions, whose derivatives of various orders are generated automatically.

- Harmonic Bonds
- Harmonic Angles
- CHARMM Torsions
- Non periodic electrostatics
- Non periodic Leonnard Jones
- Tensorfield Networks

Implementing a custom force is incredibly easy. Just define the energy function, and the underlying machinery will figure out all the necessary derivatives.

# Requirements

See requirements.txt

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