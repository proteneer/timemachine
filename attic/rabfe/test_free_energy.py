from importlib import resources

import numpy as np
from rdkit import Chem

from timemachine import constants
from timemachine.constants import DEFAULT_FF
from timemachine.fe import estimator, free_energy, topology
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.parallel.client import CUDAPoolClient

def test_absolute_free_energy():
    np.random.seed(2022)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
    mol = all_mols[1]

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(str(path_to_pdb))

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    ff_params = ff.get_ordered_params()

    seed = 2021

    lambda_schedule = np.linspace(0, 1.0, 4)
    equil_steps = 1000
    prod_steps = 1000

    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)

    def absolute_model(ff_params):

        dGs = []

        for host_system, host_coords, host_box in [
            (complex_system, complex_coords, complex_box),
            (solvent_system, solvent_coords, solvent_box),
        ]:

            # minimize the host to avoid clashes
            host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, ff, host_box)

            unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff_params, host_system)
            coords = afe.prepare_combined_coords(host_coords)
            harmonic_bond_potential = unbound_potentials[0]
            group_idxs = get_group_indices(get_bond_list(harmonic_bond_potential))

            x0 = coords
            v0 = np.zeros_like(coords)
            client = CUDAPoolClient(1)
            temperature = 300.0
            pressure = 1.0
            beta = 1 / (constants.BOLTZ * temperature)
            endpoint_correct = False

            integrator = LangevinIntegrator(temperature, 1.5e-3, 1.0, masses, seed)

            barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_idxs, 25, seed)

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                endpoint_correct,
                client,
                host_box,
                x0,
                v0,
                integrator,
                barostat,
                lambda_schedule,
                equil_steps,
                prod_steps,
                beta,
                "prefix",
            )

            dG, _, _ = estimator.deltaG(model, sys_params, subsample_interval=10)
            dGs.append(dG)

        return dGs[0] - dGs[1]

    dG = absolute_model(ff_params)
    assert np.abs(dG) < 1000.0