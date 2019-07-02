import os
import numpy as np

from system import serialize
from system import forcefield
from system import simulation

from rdkit import Chem

from simtk import openmm as mm
from simtk.openmm import app

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers


def alchemical_transform(
    scale_matrix,
    lamb,
    offset):
    """
    Alchemically transform the scale matrix.

    Parameters
    ----------
    scale_matrix: [N, N] np.array
        scale_matrix used to scale various interactions

    lamb: float
        lambda use to adjust the intermolecular interactions

    offset: int
        where to cut off the interactions

    Returns
    -------
    [N, N] np.array
        Alchemically scaled interactions

    """
    lambda_scale_matrix = np.copy(scale_matrix)
    lambda_scale_matrix[:offset, offset:] *= lamb
    lambda_scale_matrix[offset:, :offset] *= lamb
    return lambda_scale_matrix

def run_simulation():

    filepath = 'examples/host_acd.xml'
    filename, file_extension = os.path.splitext(filepath)
    sys_xml = open(filepath, 'r').read()
    system = mm.XmlSerializer.deserialize(sys_xml)
    coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    coords = coords/10

    host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, coords)

    guest_mol2 = open("/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2/guest-1.mol2", "r").read()
    mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, smirnoff_params,
        host_param_groups, smirnoff_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    es_lambda_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lj_lambda_schedule = [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.0]

    num_host_atoms = host_conf.shape[0]

    def filter_groups(param_groups, groups):
        roll = np.zeros_like(param_groups)
        for g in groups:
            roll = np.logical_or(roll, param_groups == g)
        return roll

    res = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)

    # host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)
    # guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7])).reshape(-1)
    # combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7])).reshape(-1)

    host_dp_idxs = np.array([1])
    guest_dp_idxs = np.array([1])
    combined_dp_idxs = np.array([1])

    RH = simulation.run_simulation(
        host_potentials,
        host_params,
        host_param_groups,
        host_conf,
        host_masses,
        host_dp_idxs,
        1000,
        None
    )

    H_E, H_derivs, _ = simulation.average_E_and_derivatives(RH) # [host_dp_idxs,]

    RG = simulation.run_simulation(
        guest_potentials,
        smirnoff_params,
        smirnoff_param_groups,
        guest_conf,
        guest_masses,
        guest_dp_idxs,
        1000,
        None
    )

    G_E, G_derivs, _ = simulation.average_E_and_derivatives(RG) # [guest_dp_idxs,]

    print("Noninteracting Host-Guest", H_E + G_E)

    RHG = simulation.run_simulation(
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        1000,
        None
    )

    HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG) # [combined_dp_idxs,]

    print("Interacting Host-Guest", HG_E)


    for es_lambda, lj_lambda in zip(es_lambda_schedule, lj_lambda_schedule):
        all_ps = []
        for p in combined_potentials:

            fn = p[0]
            if fn == custom_ops.Electrostatics_f32:
                nsm = alchemical_transform(p[1][0], es_lambda, num_host_atoms)
                all_ps.append((fn, (nsm, p[1][1])))
            elif fn == custom_ops.LennardJones_f32:
                nsm = alchemical_transform(p[1][0], lj_lambda, num_host_atoms)
                all_ps.append((fn, (nsm, p[1][1])))
            else:
                all_ps.append(p)


        # guest or combined? we have much bigger sensitivity if we do guest only since the counts
        # have much more flexibility
        # counts = atom_counts(guest_masses)
        # atomization_energy = np.sum(counts * linear_params)

        RHG = simulation.run_simulation(
            all_ps,
            combined_params,
            combined_param_groups,
            combined_conf,
            combined_masses,
            combined_dp_idxs,
            1000,
            None
        )

        HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG) # [combined_dp_idxs,]

        print(es_lambda, lj_lambda)


        # print(all_ps)

run_simulation()