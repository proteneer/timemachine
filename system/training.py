import os
import sys
import numpy as np

from scipy import stats
from rdkit import Chem

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.lib import custom_ops
from jax.experimental import optimizers


def run_simulation(host_params, host_steps, smirnoff_params, guest_steps, combined_params, combined_steps, prev_loss):
    
    global true_free_energy

    RH = simulation.run_simulation(
        host_potentials,
        host_params,
        host_param_groups,
        host_conf,
        host_masses,
        host_dp_idxs,
        1000,
        host_steps
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
        guest_steps,
    )

    G_E, G_derivs, _ = simulation.average_E_and_derivatives(RG) # [guest_dp_idxs,]

    RHG = simulation.run_simulation(
        combined_potentials,
        combined_params,
        combined_param_groups,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        1000,
        combined_steps,
    )

    HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG) # [combined_dp_idxs,]
    
    pred_enthalpy = HG_E - (G_E + H_E)
    delta = pred_enthalpy - true_free_energy
    num_atoms = len(combined_masses)
    loss = delta**2
    
    # fancy index into the full derivative set
    combined_derivs = np.zeros_like(combined_params)
    # remember its HG - H - G
    combined_derivs[combined_dp_idxs] += HG_derivs
    combined_derivs[host_dp_idxs] -= H_derivs
    combined_derivs[guest_dp_idxs + len(host_params)] -= G_derivs
    # combined_derivs = 2*delta*combined_derivs # L2 derivative
    combined_derivs = (delta/np.abs(delta))*combined_derivs # L1 derivative
    prev_loss = loss

    host_derivs = combined_derivs[:len(host_params)]
    guest_derivs = combined_derivs[len(host_params):]
    
    print("---------------------True, Pred, Loss", true_free_energy, pred_enthalpy, loss)
    file.write("---------------------True, Pred, Loss {} {} {}\n".format(true_free_energy, pred_enthalpy, loss))
    data.write("---------------------True, Pred, Loss {} {} {}\n".format(true_free_energy, pred_enthalpy, loss))

    return host_derivs, guest_derivs, combined_derivs, pred_enthalpy, prev_loss

# res = run_simulation(host_params, smirnoff_params, combined_params)
# return res[-1]

expected = {'guest-1.mol2':-2.17*4.184,
'guest-2.mol2':-4.19*4.184,
'guest-3.mol2':-5.46*4.184,
'guest-4.mol2':-2.74*4.184,
'guest-5.mol2':-2.99*4.184,
'guest-6.mol2':-2.53*4.184,
'guest-7.mol2':-3.4*4.184,
'guest-8.mol2':-4.89*4.184,
'guest-s9.mol2':-2.57*4.184,
'guest-s10.mol2':-2.68*4.184,
'guest-s11.mol2':-3.28*4.184,
'guest-s12.mol2':-4.2*4.184,
'guest-s13.mol2':-4.28*4.184,
'guest-s14.mol2':-4.66*4.184,
'guest-s15.mol2':-4.74*4.184,
'guest-s16.mol2':-2.75*4.184,
'guest-s17.mol2':-0.93*4.184,
'guest-s18.mol2':-2.75*4.184,
'guest-s19.mol2':-4.12*4.184,
'guest-s20.mol2':-3.36*4.184,
'guest-s21.mol2':-4.19*4.184,
'guest-s22.mol2':-4.48*4.184,}

base_dir = "/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2"

data = open("/home/yutong/Code/timemachine/system/data_adam.csv","a+")
file = open("/home/yutong/Code/timemachine/system/output_adam.txt","a+")
plot = open("/home/yutong/Code/timemachine/system/plot_adam.csv","a+")
loss_plot = open("/home/yutong/Code/timemachine/system/loss_plot_adam.csv","a+")

# files = np.array(['guest-s12.mol2', 'guest-s14.mol2', 'guest-s16.mol2', 'guest-s20.mol2', 'guest-s19.mol2', 'guest-2.mol2', 'guest-4.mol2', 'guest-s22.mol2', 'guest-6.mol2', 'guest-5.mol2', 'guest-s11.mol2', 'guest-3.mol2', 'guest-8.mol2', 'guest-s17.mol2', 'guest-s10.mol2', 'guest-s21.mol2', 'guest-7.mol2', 'guest-s13.mol2', 'guest-1.mol2', 'guest-s18.mol2', 'guest-s9.mol2', 'guest-s15.mol2'])

files = np.array(['guest-1.mol2', 'guest-2.mol2'])

host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system('/home/yutong/Code/timemachine/examples/host_acd.xml')

# setting general smirnoff parameters for guest
sdf_file = open('/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2/guest-1.mol2','r').read()
mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
            host_potentials, guest_potentials,
            host_params, smirnoff_params,
            host_param_groups, smirnoff_param_groups,
            host_conf, guest_conf,
            host_masses, guest_masses)

def filter_groups(param_groups, groups):
    roll = np.zeros_like(param_groups)
    for g in groups:
        roll = np.logical_or(roll, param_groups == g)
    return roll

host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7,4,5,0,1,2,3])).reshape(-1)
guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7,4,5,0,1,2,3])).reshape(-1)
combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [7,4,5,0,1,2,3])).reshape(-1)

lr = .0005

plot.write('epoch,guest,true enthalpy,computed enthalpy\n')
loss_plot.write('epoch,r2,mae,loss\n')

prev_loss = None

count = 0

init_params = combined_params
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(init_params)

for t in range(35):
    print('epoch:',t)
    
    np.random.shuffle(files)
    
    x = np.array([])
    y = np.array([])
    errors = np.array([])
    losses = np.array([])
    
    for filename in files:
        
        pred_enthalpy = None
        
        true_free_energy = expected[filename]

        sdf_file = open('/home/yutong/Code/benchmarksets/input_files/cd-set1/mol2/' + filename,'r').read()
        mol = Chem.MolFromMol2Block(sdf_file, sanitize=True, removeHs=False, cleanupSubstructures=True)

        guest_potentials, _, _, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

        combined_potentials, _, _, combined_conf, combined_masses = forcefield.combiner(
            host_potentials, guest_potentials,
            host_params, smirnoff_params,
            host_param_groups, smirnoff_param_groups,
            host_conf, guest_conf,
            host_masses, guest_masses)

        print('--------------------running simulation for {} in host-acd, temp = 5 K, all charges & torsions, alpha=.0005--------------------'.format(filename))
        file.write('---------------------running simulation for {} in host-acd, temp = 5 K, all charges & torsions, alpha=.0005--------------------\n'.format(filename))
        data.write('---------------------running simulation for {} in host-acd, temp = 5 K, all charges & torsions, alpha=.0005--------------------\n'.format(filename))

        # try:        
            # for t in range(50):
            
            # opt_state, pred_enthalpy, prev_loss = step(count, opt_state)
            # def step(i, opt_state):   
                # global prev_loss, combined_params, host_params, smirnoff_params
            
        combined_params = get_params(opt_state)
        host_params = combined_params[:len(host_params)]
        smirnoff_params = combined_params[len(host_params):]
        
        np.set_printoptions(suppress=True)
        print("current_params, {}".format(combined_params[combined_dp_idxs]))
        file.write('current_params, {}\n'.format(combined_params[combined_dp_idxs]))
        
        host_derivs, guest_derivs, combined_derivs, pred_enthalpy, prev_loss = run_simulation(host_params, 15000, smirnoff_params, 7500, combined_params, 30000, prev_loss)
        plot.write('{},{},{},{}\n'.format(t,filename,true_free_energy,pred_enthalpy))
        
        opt_state = opt_update(count,combined_derivs,opt_state)

        # return opt_update(i, combined_derivs, opt_state), pred_enthalpy, prev_loss
        
        x = np.append(x,true_free_energy)
        y = np.append(y,pred_enthalpy)
        errors = np.append(errors, abs(true_free_energy - pred_enthalpy))
        losses = np.append(losses, prev_loss)
        
        count += 1

        # except Exception as e:
        #     converged = 'fail'
        #     print("simulation failed {} for {}".format(e, filename))
        #     file.write("simulation failed {} for {}\n".format(e, filename))

    r2 = (stats.pearsonr(x,y)[0])**2
    mae = np.mean(errors)
    average_loss = np.mean(losses)
    data.write('r^2 for epoch {}: {}\n'.format(t,r2))
    data.write('MAE for epoch {}: {}\n'.format(t,mae))
    loss_plot.write('{},{},{}\n'.format(t,r2,mae,average_loss))
    print('r^2 for epoch {}: {}\n'.format(t,r2))
    print('MAE for epoch {}: {}\n'.format(t,mae))
            

            # Gradient Descent Optimizer
    #         def gd_opt(lr):
    #             for _ in range(25):
    #                 # print("current_params", combined_params[combined_dp_idxs])
    #                 print('learning rate: {}'.format(lr))
    #                 np.set_printoptions(suppress=True)
    #                 file.write("current_params, {}\n".format(combined_params[combined_dp_idxs]))
    #                 host_derivs, guest_derivs, combined_derivs, pred_enthalpy, prev_loss = run_simulation(host_params, 5000, smirnoff_params, 5000, combined_params, 5000, prev_loss)
    #                 host_params -= lr*host_derivs
    #                 smirnoff_params -= lr*guest_derivs
    #                 combined_params -= lr*combined_derivs
    #                 np.testing.assert_almost_equal(np.concatenate([host_params, smirnoff_params]), combined_params)
    #                 if (_ - 1) % 4 == 0:
    #                     lr = lr / 2
#             return pred_enthalpy