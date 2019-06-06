import numpy as np

from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation
from openforcefield.typing.engines.smirnoff import ForceField

from timemachine.lib import custom_ops

import os

host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system('/home/ubuntu/Relay/Code/timemachine/examples/host_systems/host-acd/host-acd.xml')

# data = open("/home/ubuntu/Relay/Code/timemachine/system/data.csv","a")

directory = "/home/ubuntu/Relay/Code/timemachine/examples/guest_systems/"

# for filename in os.listdir(directory):
#     if "_" not in filename:
mol2path = directory + "guest-s17" + "/" + "guest-s17" + ".mol2"
print(mol2path)

f = open(mol2path,"r")
test_sdf = f.read()

mol = Chem.MolFromMol2Block(test_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

guest_potentials, guest_params, guest_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
    host_potentials, guest_potentials,
    host_params, guest_params,
    host_param_groups, guest_param_groups,
    host_conf, guest_conf,
    host_masses, guest_masses)

# print(host_conf)
# print(len(host_params))
# print(len(host_param_groups))
# assert 0
for _ in range(5):
    R_host, host_converge = simulation.run_simulation(
        host_potentials,
        host_params,
        host_param_groups,
        host_conf,
        host_masses,
        np.argwhere(host_param_groups == 7).reshape(-1),
        250,
        50000
    )

    # print(len(R))

    R_guest, guest_converge = simulation.run_simulation(
        guest_potentials,
        guest_params,
        guest_param_groups,
        guest_conf,
        guest_masses,
        np.argwhere(guest_param_groups == 7).reshape(-1),
        250,
        20000
    )

    # print(len(R))

    R_comb, comb_converge = simulation.run_simulation(
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        np.argwhere(guest_param_groups == 7).reshape(-1),
        250,
        100000
    )

    # print(len(R))

    host_avg_E = np.sum(np.array(R_host)) / len(R_host) 
    guest_avg_E = np.sum(np.array(R_guest)) / len(R_guest)
    comb_avg_E = np.sum(np.array(R_comb)) / len(R_comb)
    enthalpy = comb_avg_E - (host_avg_E + guest_avg_E)
    print('{},{},{},{},{},{},{},{}\n'.format("guest-s17",host_avg_E,host_converge,guest_avg_E,guest_converge,comb_avg_E,comb_converge,enthalpy))
    
    
# print('host: {}'.format(host_avg_E))
# print('guest: {}'.format(guest_avg_E))
# print('combined: {}'.format(comb_avg_E))
