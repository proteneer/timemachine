from md.builders import build_water_system
from md.minimizer import minimize_host_4d
from md.ensembles import PotentialEnergyModel, NVTEnsemble
from timemachine.lib import LangevinIntegrator
from fe.free_energy import AbsoluteFreeEnergy

from testsystems.relative import hif2a_ligand_pair

from simtk import unit
import numpy as np

temperature = 300 * unit.kelvin
initial_waterbox_width = 3.0 * unit.nanometer
timestep = 1.5 * unit.femtosecond
collision_rate = 1.0 / unit.picosecond
seed = 2021

mol_a, _, core, ff = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core, hif2a_ligand_pair.ff
complex_system, complex_coords, complex_box, complex_top = build_water_system(
    initial_waterbox_width.value_in_unit(unit.nanometer))

min_complex_coords = minimize_host_4d([mol_a], complex_system, complex_coords, ff, complex_box)
afe = AbsoluteFreeEnergy(mol_a, ff)

unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
    ff.get_ordered_params(), complex_system, min_complex_coords
)

potential_energy_model = PotentialEnergyModel(sys_params, unbound_potentials, precision=np.float32)
integrator = LangevinIntegrator(
    temperature.value_in_unit(unit.kelvin),
    timestep.value_in_unit(unit.picosecond),
    collision_rate.value_in_unit(unit.picosecond ** -1),
    masses,
    seed
)
integrator_impl = integrator.impl()
bound_impls = potential_energy_model.all_impls

ensemble = NVTEnsemble(potential_energy_model, temperature)
