# relative hydration free energy
import os
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from fe import pdb_writer
from fe import topology
from md import builders
from md import minimizer

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

import functools
import jax

from ff import Forcefield

from ff.handlers import nonbonded, bonded, openmm_deserializer

from ff.handlers.deserialize import deserialize_handlers

from multiprocessing import Pool

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm


def run(args):

    lamb_idx, final_lamb, gpu_idx = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    suppl = Chem.SDMolSupplier('tests/data/ligands_40.sdf', removeHs=False)
    # suppl = Chem.SDMolSupplier('tests/data/benzene_flourinated.sdf', removeHs=False)
    all_mols = [x for x in suppl]

    romol_a = all_mols[1]
    romol_b = all_mols[4]

    ligand_masses_a = [a.GetMass() for a in romol_a.GetAtoms()]
    ligand_masses_b = [a.GetMass() for a in romol_b.GetAtoms()]

    # extract the 0th conformer
    ligand_coords_a = get_romol_conf(romol_a)
    ligand_coords_b = get_romol_conf(romol_b)

    # construct a 4-nanometer water box (from openmmtools approach: selecting out
    #   of a large pre-equilibrated water box snapshot)
    system, host_coords, box, omm_topology = builders.build_water_system(4.0)

    # padding to avoid jank
    box = box + np.eye(3)*0.1

    host_bps, host_masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)


    # minimize coordinates

    # note: .py file rather than .offxml file
    # note: _ccc suffix means "correctable charge corrections"
    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    # delete me later
    # ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_recharge.py').read())
    ff = Forcefield(ff_handlers)

    # for RHFE we need to insert the reference ligand first, before inserting the
    # decoupling ligand
    # print("start minimization")
    minimized_host_coords = minimizer.minimize_host_4d(romol_a, system, host_coords, ff, box)
    # print("end minimization")

    num_host_atoms = host_coords.shape[0]

    final_potentials = []
    final_vjp_and_handles = []

    # keep the bonded terms in the host the same.
    # but we keep the nonbonded term for a subsequent modification
    for bp in host_bps:
        if isinstance(bp, potentials.Nonbonded):
            host_p = bp
        else:
            final_potentials.append([bp])
            final_vjp_and_handles.append(None)

    class CompareDist(rdFMCS.MCSAtomCompare):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compare(self, p, mol1, atom1, mol2, atom2):
            x_i = mol1.GetConformer(0).GetPositions()[atom1]
            x_j = mol2.GetConformer(0).GetPositions()[atom2]
            if np.linalg.norm(x_i-x_j) > 0.3:
                return False
            else:
                return True

    # this requires a new RDKit build with the following patch applied:
    # https://github.com/rdkit/rdkit/pull/3638        
    core = None
    core = np.array([[ 0,  0],
       [ 2,  2],
       [ 1,  1],
       [ 6,  6],
       [ 5,  5],
       [ 4,  4],
       [ 3,  3],
       [15, 16],
       [16, 17],
       [17, 18],
       [18, 19],
       [19, 20],
       [20, 21],
       [32, 30],
       [26, 25],
       [27, 26],
       [ 7,  7],
       [ 8,  8],
       [ 9,  9],
       [10, 10],
       [29, 11],
       [11, 12],
       [12, 13],
       [14, 15],
       [31, 29],
       [13, 14],
       [23, 24],
       [30, 28],
       [28, 27],
       [21, 22]])

    if core is None:

        mcs_params = rdFMCS.MCSParameters()
        mcs_params.AtomTyper = CompareDist()
        mcs_params.BondCompareParameters.CompleteRingsOnly = 1
        mcs_params.BondCompareParameters.RingMatchesRingOnly = 1
        mcs_params.AtomCompareParameters.matchValences = 1
                
        res = rdFMCS.FindMCS(
            [romol_a, romol_b],
            mcs_params
        )

        core_smarts = res.smartsString
        query_mol = Chem.MolFromSmarts(core_smarts)

        core_a = romol_a.GetSubstructMatches(query_mol)
        core_b = romol_b.GetSubstructMatches(query_mol)

        assert len(core_a) == 1
        assert len(core_b) == 1

        core_a = core_a[0]
        core_b = core_b[0]

        core = np.stack([core_a, core_b], axis=1)

    # core = np.stack([
        # np.arange(romol_a.GetNumAtoms()),
        # np.arange(romol_b.GetNumAtoms())
    # ], axis=1)


    # core = core[:-1] # remove the H-F mapping, become 4D
    # core = np.concatenate([core[:7], core[8:]])

    gdt = topology.SingleTopology(romol_a, romol_b, core, ff)
    hgt = topology.HostGuestTopology(host_p, gdt)

    # setup the parameter handlers for the ligand
    bonded_tuples = [
        [hgt.parameterize_harmonic_bond, ff.hb_handle],
        [hgt.parameterize_harmonic_angle, ff.ha_handle],
        [hgt.parameterize_proper_torsion, ff.pt_handle],
        [hgt.parameterize_improper_torsion, ff.it_handle]
    ]

    # instantiate the vjps while parameterizing (forward pass)
    for fn, handle in bonded_tuples:
        (src_params, dst_params, uni_params), vjp_fn, (src_potential, dst_potential, uni_potential) = jax.vjp(fn, handle.params, has_aux=True)
        final_potentials.append([src_potential.bind(src_params), dst_potential.bind(dst_params), uni_potential.bind(uni_params)])
        final_vjp_and_handles.append((vjp_fn, handle))

    nb_params, vjp_fn, nb_potential = jax.vjp(hgt.parameterize_nonbonded, ff.q_handle.params, ff.lj_handle.params, has_aux=True)
    final_potentials.append([nb_potential.bind(nb_params)])
    final_vjp_and_handles.append([vjp_fn])

    # note: lambda goes from 0 to 1, 0 being fully-interacting and 1.0 being fully interacting.
    # for lamb_idx, final_lamb in enumerate(np.linspace(0, 1.0, 40)):
    # for lamb_idx, final_lamb in enumerate([0.05, 0.95]):
    # for lamb_idx, final_lamb in enumerate([0.0, 1.0]):


    # write some conformations into this PDB file
    writer = pdb_writer.PDBWriter([omm_topology, romol_a, romol_b], "debug_"+str(lamb_idx)+".pdb")

    seed = np.random.randint(0, 501237509)

    # note: OpenMM unit system used throughout
    #   (temperature: kelvin, timestep: picosecond, collision_rate: picosecond^-1)
    combined_masses = np.concatenate([host_masses, np.mean(gdt.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)])

    intg = LangevinIntegrator(
        300.0,
        1.5e-3,
        1.0,
        combined_masses,
        seed
    ).impl()

    # note the order in which the coordinates are concatenated in this step --
    #   in a later step we will need to combine recipes in the same order
    src_conf, dst_conf = gdt.interpolate_params(ligand_coords_a, ligand_coords_b)
    combined_coords = np.concatenate([minimized_host_coords, np.mean(gdt.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)])

    # re-interpolate the coordinates from original 
    x0 = combined_coords
    v0 = np.zeros_like(x0)

    u_impls = []
    bonded_impls = []
    nonbonded_impls = []

    for bps in final_potentials:
        for bp in bps:
            impl = bp.bound_impl(np.float32)
            if isinstance(bp, potentials.InterpolatedPotential):
                nonbonded_impls.append(impl)
            elif isinstance(bp, potentials.LambdaPotential):
                bonded_impls.append(impl)
            u_impls.append(impl)

    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    # (ytz): split the coordinates back out
    def unjank_x(x):
        c_x = x[num_host_atoms:]
        a_x = [None]*romol_a.GetNumAtoms()
        b_x = [None]*romol_b.GetNumAtoms()
        for a, c in enumerate(gdt.a_to_c):
            a_x[a] = c_x[c]
        for b, c in enumerate(gdt.b_to_c):
            b_x[b] = c_x[c]

        return np.concatenate([
            x[:num_host_atoms],
            a_x,
            b_x
        ])

    # for step, lamb in enumerate(np.linspace(1.0, final_lamb, 1000)):
    #     if step % 500 == 0:
    #         writer.write_frame(unjank_x(ctxt.get_x_t())*10)
    #     ctxt.step(lamb)
        # print(step, ctxt.get_u_t())

    # print("insertion energy", ctxt.get_u_t())
    for step in range(10000):
        if step % 500 == 0:
            writer.write_frame(unjank_x(ctxt.get_x_t())*10)
        ctxt.step(final_lamb)

    # print("equilibrium energy", ctxt.get_u_t())

    bonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(bonded_impls, 5)
    nonbonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(nonbonded_impls, 5)

    ctxt.add_observable(bonded_du_dl_obs)
    ctxt.add_observable(nonbonded_du_dl_obs)

    du_dps = []
    for ui in u_impls:
        du_dp_obs = custom_ops.AvgPartialUPartialParam(ui, 5)
        ctxt.add_observable(du_dp_obs)
        du_dps.append(du_dp_obs)


    for _ in range(100000):
        if step % 500 == 0:
            writer.write_frame(unjank_x(ctxt.get_x_t())*10)
        ctxt.step(final_lamb)

    # print("final_nrg", ctxt.get_u_t())

    writer.close()

    # print("final energy", ctxt.get_u_t())

    # print vector jacobian products back into the forcefield derivative
    # for du_dp_obs, vjp_and_handles in zip(du_dps, final_vjp_and_handles):
    #     du_dp = du_dp_obs.avg_du_dp()

    #     if vjp_and_handles:
    #         vjp_fn, handles = vjp_and_handles
    #         du_df = vjp_fn(du_dp) # vjp into forcefield derivatives
    #         for f_grad, h in zip(du_df, handles):
    #             print("handle:", type(h).__name__)
    #             for s, vv in zip(h.smirks, f_grad):
    #                 if np.any(vv) != 0:
    #                     print(s, vv)
    #             print("\n")

    print("lambda", final_lamb, "bonded du_dl", bonded_du_dl_obs.avg_du_dl(), "nonbonded du_dl", nonbonded_du_dl_obs.avg_du_dl())

    return bonded_du_dl_obs.avg_du_dl(), nonbonded_du_dl_obs.avg_du_dl()
    # assert 0

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="RBFE testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus"
    )

    cmd_args = parser.parse_args()

    p = Pool(cmd_args.num_gpus)

    run_args = []
    for lamb_idx, final_lamb in enumerate(np.linspace(0, 1.0, 50)):
        run_args.append((lamb_idx, final_lamb, lamb_idx % cmd_args.num_gpus))

    res = p.map(run, run_args, chunksize=1)