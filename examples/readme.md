### Examples

#### Absolute hydration free energy

`ahfe.py` runs 4D decoupling simulations of aspirin in a waterbox, for later use in a TI estimator of the absolute hydration free energy.

#### Relative hydration free energy

* `rhfe_dual.py` runs 4D decoupling simulations of aspirin and a version of aspirin that mutates an oxygen to a fluorine in a waterbox, subject to a centroid restraint, for later use in a TI estimator of the relative hydration free energy difference between these two compounds.
* `rhfe_single.py` runs three hydration free energy protocols that should give similar results:
    * taking the difference between two absolute hydration free energy calculations (each computed using TI and 4D decoupling)
    * computing a relative free energy using a complete atom-mapping (interpolating the valence energies, and interpolating the nonbonded parameters)
    * computing a relative free energy using a partial atom-mapping (the atom being deleted / inserted is handled using 4D decoupling)

#### Relative binding free energy

* `rbfe_single.py` runs single-topology calculations for a single pair of ligands for the HIF2A benchmark system, and adjusts the charges and Lennard-Jones sigma parameters to improve fit to reference values
* `hif2a/` runs calculations on several of the ligands in the hif2a benchmark dataset
  * `hif2a/generate_star_map.py` generates a "star map" (a single hub and several spokes), and serializes a collection of RelativeFreeEnergy objects for (hub, spoke) pairs
  * `hif2a/fit_to_multiple_rbfes.py` fits nonbonded parameters to the experimental IC50s associated with these ligands

#### Testing overlap

`overlap_test.py` tests two restraint types, one based on principal moments of inertia and one based on a measure of shape overlap.

#### Measuring performance

`estimator_variance.py` measures run-to-run variability as a function of number of MD steps performed per run