# This script generates vanilla trajectories of compounds in water and vacuum.

import argparse
import multiprocessing
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe import estimator, model_rabfe, model_utils, topology
from timemachine.fe.free_energy import AbsoluteFreeEnergy, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list
from timemachine.parallel.client import CUDAPoolClient
from timemachine.parallel.utils import get_gpu_count


def smiles_file_to_mols(path):
    smi_params = Chem.SmilesParserParams()
    smi_params.removeHs = False
    with open(path, "r") as ifs:
        for line in ifs.readlines():
            if line.startswith("SMILES"):
                continue
            parts = line.strip("\n").split(" ")
            smi_string = parts[0]
            name = " ".join(parts[1:])
            mol = Chem.MolFromSmiles(smi_string, smi_params)
            mol.SetProp("_Name", name)
            yield mol


def simulate_solvent(model, mol, host_coords, box, seed):
    mol_coords = get_romol_conf(mol)
    minimized_host_coords = minimizer.minimize_host_4d(
        [mol],
        model.host_system,
        host_coords,
        model.ff,
        box,
        [mol_coords],
    )
    ordered_params = model.ff.get_ordered_params()
    topo = model.setup_topology(mol)
    afe = AbsoluteFreeEnergy(mol, topo)
    unbound_pots, sys_params, _ = afe.prepare_host_edge(ordered_params, model.host_system)
    bound_pots = []
    for params, unbound_pot in zip(sys_params, unbound_pots):
        bound_pots.append(unbound_pot.bind(params))
    return bound_pots, model.simulate_futures(
        ordered_params,
        mol,
        np.concatenate([minimized_host_coords, mol_coords]),
        box,
        prefix="solvent_" + mol.GetProp("_Name"),
        seed=seed,
    )


def simulate_vacuum(client, mol, ff, temperature, dt, equil_steps, prod_steps, seed):
    mol_coords = get_romol_conf(mol)
    top = topology.BaseTopology(mol, ff)
    ordered_params = ff.get_ordered_params()
    afe = AbsoluteFreeEnergy(mol, top)

    unbound_potentials, sys_params, masses = afe.prepare_vacuum_edge(ordered_params)

    if seed == 0:
        seed = np.random.randint(np.iinfo(np.int32).max)

    if dt > 1.5e-3:
        bond_list = get_bond_list(unbound_potentials[0])
        masses = model_utils.apply_hmr(masses, bond_list)
    friction = 1.0
    integrator = LangevinIntegrator(temperature, dt, friction, masses, seed)

    box = np.eye(3) * 1000

    v0 = np.zeros_like(mol_coords)

    barostat = None

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)
    subsample_interval = min(prod_steps, 1000)
    lamb = 0.0  # Fully embedded ligand
    lambda_schedule = [lamb]
    args = (
        lamb,
        box,
        mol_coords,
        v0,
        bound_potentials,
        integrator,
        barostat,
        equil_steps,
        prod_steps,
        subsample_interval,
        subsample_interval,
        lambda_schedule,
    )
    futures = []
    futures.append(client.submit(estimator.simulate, *args))
    return bound_potentials, futures


def main():
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Simulate compounds from SMILES in Vacuum/Solvent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("smiles_file", help="Path to Smiles")

    parser.add_argument("--limit", default=None, type=int, help="Number of compounds to simulate")

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        default=100_000,
        help="number of equilibration steps for solvent/vacuum",
    )

    parser.add_argument(
        "--num_prod_steps", type=int, help="number of production steps for each simulation", default=1_000_000
    )

    parser.add_argument("--seed", default=2022, type=int)

    cmd_args = parser.parse_args()

    seed = cmd_args.seed
    num_gpus = cmd_args.num_gpus
    # set up multi-GPU client
    client = CUDAPoolClient(max_workers=num_gpus)
    client.verify()

    lambda_schedule = np.zeros(1)
    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-5
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    solvent_model = model_rabfe.AbsoluteHydrationModel(
        client,
        forcefield,
        solvent_system,
        lambda_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
    )

    count = 0
    futures = []
    for mol in smiles_file_to_mols(cmd_args.smiles_file):
        if cmd_args.limit is not None and count >= cmd_args.limit:
            break
        assert mol.GetNumAtoms() == AllChem.AddHs(mol).GetNumAtoms()
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        sim_futures = simulate_solvent(solvent_model, mol, solvent_coords, solvent_box, seed)
        sim_vacuum_futures = simulate_vacuum(
            client, mol, forcefield, temperature, dt, cmd_args.num_equil_steps, cmd_args.num_prod_steps, seed
        )

        futures.append((mol, sim_futures, sim_vacuum_futures))
        count += 1
    for mol, (solvent_pots, (_, free_energy, solv_futures)), (vacuum_pots, vacuum_futures) in futures:
        mol_name = mol.GetProp("_Name")
        assert len(solv_futures) == 1
        assert len(vacuum_futures) == 1
        with Chem.PDBWriter(f"vacuum_topology_{mol_name}.pdb") as writer:
            writer.write(mol)
        model_utils.generate_openmm_topology(
            [solvent_model.host_topology, mol],
            free_energy.x0,
            box=free_energy.box,
            out_filename=f"solvent_topology_{mol_name}.pdb",
        )
        with open(f"solvent_potentials_{mol_name}.pkl", "wb") as ofs:
            pickle.dump(solvent_pots, ofs)
        with open(f"vacuum_potentials_{mol_name}.pkl", "wb") as ofs:
            pickle.dump(vacuum_pots, ofs)
        results = [f.result() for f in solv_futures]
        for res in results:
            np.savez(
                f"solvent_{mol_name}_frames.npz",
                xs=res.xs,
                boxes=res.boxes,
                lambda_us=res.lambda_us,
            )
        results = [f.result() for f in vacuum_futures]
        for res in results:
            np.savez(
                f"vacuum_{mol_name}_frames.npz",
                xs=res.xs,
                boxes=res.boxes,
                lambda_us=res.lambda_us,
            )


if __name__ == "__main__":
    main()
