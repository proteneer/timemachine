### Examples

#### MD
`npt_equilibration.py` samples the constant temperature and pressure ensemble for each of several lambda windows, by running thermostatted MD and pausing every few steps to apply a Monte Carlo barostat move

#### Absolute hydration free energy

`ahfe.py` runs 4D decoupling simulations of aspirin in a waterbox, for later use in a TI estimator of the absolute hydration free energy.

#### Relative hydration free energy

* `rhfe_dual.py` runs 4D decoupling simulations of aspirin and a version of aspirin that mutates an oxygen to a fluorine in a waterbox, subject to a centroid restraint, for later use in a TI estimator of the relative hydration free energy difference between these two compounds.

#### Relative binding free energy

* `hif2a/` runs calculations on several of the ligands in the hif2a benchmark dataset
  * `hif2a/generate_star_map.py` generates a "star map" (a single hub and several spokes), and serializes a collection of RelativeFreeEnergy objects for (hub, spoke) pairs
  * `hif2a/fit_to_multiple_rbfes.py` fits nonbonded parameters to the experimental IC50s associated with these ligands

#### Testing overlap

`overlap_test.py` tests two restraint types, one based on principal moments of inertia and one based on a measure of shape overlap.

#### Interfacing with Jax
`potential_energy.py` computes the potential energy of an "alchemical" system, as well as its derivatives w.r.t. coords, params, or lam
