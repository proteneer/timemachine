import os
import sys
import numpy as np

from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation
from openforcefield.typing.engines.smirnoff import ForceField

from timemachine.lib import custom_ops

def run_system(sdf_file):

    host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system('examples/host_acd.xml')

    mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, guest_params, guest_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, guest_params,
        host_param_groups, guest_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    # host_dp_idxs = np.argwhere(host_param_groups == 7).reshape(-1)
    # guest_dp_idxs = np.argwhere(guest_param_groups == 7).reshape(-1)
    # combined_dp_idxs = np.argwhere(combined_param_groups == 7).reshape(-1)

    host_dp_idxs = np.argwhere(np.logical_or(host_param_groups == 7, host_param_groups == 5)).reshape(-1)
    guest_dp_idxs = np.argwhere(np.logical_or(guest_param_groups == 7, guest_param_groups == 5)).reshape(-1)
    combined_dp_idxs = np.argwhere(np.logical_or(combined_param_groups == 7, combined_param_groups == 5)).reshape(-1)

    def run_simulation(host_params, guest_params, combined_params):

        RH = simulation.run_simulation(
            host_potentials,
            host_params,
            host_param_groups,
            host_conf,
            host_masses,
            host_dp_idxs,
            500,
            25000
        )

        H_E, H_analytic_derivs, H_thermo_derivs = simulation.average_E_and_derivatives(RH) # [host_dp_idxs,]

        RG = simulation.run_simulation(
            guest_potentials,
            guest_params,
            guest_param_groups,
            guest_conf,
            guest_masses,
            guest_dp_idxs,
            500,
            25000
        )

        G_E, G_analytic_derivs, G_thermo_derivs = simulation.average_E_and_derivatives(RG) # [guest_dp_idxs,]

        RHG = simulation.run_simulation(
            combined_potentials,
            combined_params,
            combined_param_groups,
            combined_conf,
            combined_masses,
            combined_dp_idxs,
            500,
            25000
        )

        HG_E, HG_analytic_derivs, HG_thermo_derivs = simulation.average_E_and_derivatives(RHG) # [combined_dp_idxs,]

        pred_enthalpy = HG_E - (G_E + H_E)
        # true_enthalpy = -20 # kilojoules
        true_enthalpy = 50 # kilojoules
        delta_enthalpy = pred_enthalpy - true_enthalpy
        num_atoms = combined_conf.shape[0]

        loss = delta_enthalpy**2/num_atoms

        print("-----------True, Pred, Loss (in kcal/mol)", true_enthalpy, pred_enthalpy, np.abs(delta_enthalpy)/4.184)
        # fancy index into the full derivative set
        combined_derivs = np.zeros_like(combined_params)
        # remember its HG - (H + G)
        combined_derivs[combined_dp_idxs] += HG_analytic_derivs
        combined_derivs[host_dp_idxs] -= H_analytic_derivs
        combined_derivs[guest_dp_idxs + len(host_params)] -= G_analytic_derivs
        combined_derivs = (2*delta_enthalpy*combined_derivs)/num_atoms

        host_derivs = combined_derivs[:len(host_params)]
        guest_derivs = combined_derivs[len(host_params):]

        return host_derivs, guest_derivs, combined_derivs

    for epoch in range(500):

        print("starting epoch", epoch)
        # print("current_params", epoch, combined_params[combined_dp_idxs])

        host_derivs, guest_derivs, combined_derivs = run_simulation(host_params, guest_params, combined_params)
        lr = 1e-6
        host_params -= lr*host_derivs
        guest_params -= lr*guest_derivs
        combined_params -= lr*combined_derivs


    return res[-1]

base_dir = "/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2"

results = []
for filename in sorted(os.listdir(base_dir)):
    if filename != 'guest-1.mol2':
        continue
    if 'guest' in filename:

        file_path = os.path.join(base_dir, filename)
        file_data = open(file_path, "r").read()
        print("processing", filename)
        pred_enthalpy = run_system(file_data)
        print(filename, pred_enthalpy)
        results.append([filename, pred_enthalpy])

# print(results)

# for _ in range(20):

#     print("current_params", combined_params[combined_dp_idxs])

#     host_derivs, guest_derivs, combined_derivs = run_simulation(host_params, guest_params, combined_params)
#     lr = 1e-6
#     host_params -= lr*host_derivs
#     guest_params -= lr*guest_derivs
#     combined_params -= lr*combined_derivs



#     np.testing.assert_almost_equal(np.concatenate([host_params, guest_params]), combined_params)

#     # host_params
#     # guest_params 