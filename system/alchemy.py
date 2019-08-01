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

import scipy.integrate
# X = [1, 2, 3, 4, 5]
# velocity = it.cumtrapz(X,initial=0)
# location = it.cumtrapz(velocity,initial=0)


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

    guest_mol2 = open("/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2/guest-2.mol2", "r").read()
    mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, smirnoff_params,
        host_param_groups, smirnoff_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    # es_lambda_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # lj_lambda_schedule = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.0]
    # es_lambda_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    # lj_lambda_schedule = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # es_lambda_schedule = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # lj_lambda_schedule = [1.0, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1, 0.05]




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

    RH, h_x = simulation.run_simulation(
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

    RG, g_x = simulation.run_simulation(
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

    print("Non Interacting Host-Guest", H_E + G_E)

    RHG, hg_x = simulation.run_simulation(
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

    print("Fully Interacting Host-Guest", HG_E)


    # es_lambda_schedule = np.linspace(1,0,12)
    # lj_lambda_schedule = np.ones_like(lj_lambda_schedule)

    # es_lambda_schedule = np.zeros_like(lj_lambda_schedule)
    # lj_lambda_schedule = np.linspace(1,0.05,12)


    #####
    # all_es_lambda_schedules = [
    #     np.linspace(1,0,100),
    #     np.zeros_like(np.linspace(1,0.0,200))
    # ]

    # all_lj_lambda_schedules = [
    #     np.ones_like(np.linspace(1,0,100)),
    #     np.linspace(1,0.0,200)
    # ]

    #####
    all_es_lambda_schedules = [
        np.linspace(1,0.0,200),
    ]

    all_lj_lambda_schedules = [
        np.linspace(1,0.0,200)
    ]


    all_dGs = []

    for idx, (es_lambda_schedule, lj_lambda_schedule) in enumerate(zip(all_es_lambda_schedules, all_lj_lambda_schedules)):


        print("es_lambda_schedule", es_lambda_schedule)
        print("lj_lambda_schedule", lj_lambda_schedule)

        dGs = []
        for es_lambda, lj_lambda in zip(es_lambda_schedule, lj_lambda_schedule):


            es_start = es_lambda_schedule[0]
            es_end = es_lambda_schedule[-1]

            lj_start = lj_lambda_schedule[0]
            lj_end = lj_lambda_schedule[-1]

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

            RHG, final_conf = simulation.run_simulation(
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

            dG = 0

            for p in combined_potentials:
                fn = p[0]
                if fn == custom_ops.Electrostatics_f32:
                    u_a_sm = alchemical_transform(p[1][0], es_start, num_host_atoms)
                    u_b_sm = alchemical_transform(p[1][0], es_end, num_host_atoms)
                    u_a = custom_ops.Electrostatics_f32(u_a_sm.astype(np.float32), p[1][1])
                    u_b = custom_ops.Electrostatics_f32(u_b_sm.astype(np.float32), p[1][1])
                    res_a = u_a.derivatives(final_conf.reshape(1,-1,3).astype(np.float32), combined_params.astype(np.float32), combined_dp_idxs.astype(np.int32))
                    res_b = u_b.derivatives(final_conf.reshape(1,-1,3).astype(np.float32), combined_params.astype(np.float32), combined_dp_idxs.astype(np.int32))
                    # print("electrostatics", res_b[0]-res_a[0])
                    dG += (res_b[0]-res_a[0])[0]
                elif fn == custom_ops.LennardJones_f32:
                    u_a_sm = alchemical_transform(p[1][0], lj_start, num_host_atoms)
                    u_b_sm = alchemical_transform(p[1][0], lj_end, num_host_atoms)
                    u_a = custom_ops.LennardJones_f32(u_a_sm.astype(np.float32), p[1][1])
                    u_b = custom_ops.LennardJones_f32(u_b_sm.astype(np.float32), p[1][1])
                    res_a = u_a.derivatives(final_conf.reshape(1,-1,3).astype(np.float32), combined_params.astype(np.float32), combined_dp_idxs.astype(np.int32))
                    res_b = u_b.derivatives(final_conf.reshape(1,-1,3).astype(np.float32), combined_params.astype(np.float32), combined_dp_idxs.astype(np.int32))
                    # print("lennardjones", res_b[0]-res_a[0])
                    dG += (res_b[0]-res_a[0])[0]


            print(es_lambda, lj_lambda, dG)

            dGs.append(dG)

        dx = 1/len(es_lambda_schedule)
        print("integral", scipy.integrate.cumtrapz(dGs, dx=dx), "@ dx:", dx)

        all_dGs.append(dGs)



    np.save("TI_2.npy", all_dGs)
        # print("dGs", dGs)



        # print(all_ps)

run_simulation()
