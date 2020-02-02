import argparse
import time
import numpy as np
from io import StringIO
import itertools
import gnuplotlib as gp
import os

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)


from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils

import multiprocessing
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation
from fe import loss


def com(conf):
    return np.sum(conf, axis=0)/conf.shape[0]

def recenter(conf, true_com, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 


from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_sort(conf):
    hc = HilbertCurve(16, 3)
    int_confs = (conf*1000).astype(np.int64)+10000
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm

class PDBWriter():

    def __init__(self, pdb_str, out_filepath):
        self.pdb_str = pdb_str
        self.out_filepath = out_filepath
        self.outfile = None
        self.n_frames = 100

    def write_header(self):
        """
        Confusingly this initializes writer as well because 
        """
        outfile = open(self.out_filepath, 'w')
        self.outfile = outfile
        cpdb = app.PDBFile(self.pdb_str)
        PDBFile.writeHeader(cpdb.topology, self.outfile)
        self.topology = cpdb.topology
        self.frame_idx = 0

    def write(self, x):
        if self.outfile is None:
            raise ValueError("remember to call write_header first")
        self.frame_idx += 1
        PDBFile.writeModel(self.topology, x, self.outfile, self.frame_idx)

    def close(self):
        PDBFile.writeFooter(self.topology, self.outfile)
        self.outfile.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_pdb', type=str)
    parser.add_argument('--complex_pdb', type=str)
    parser.add_argument('--solvent_pdb', type=str)
    parser.add_argument('--ligand_sdf', type=str)
    args = parser.parse_args()

    amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')

    # host_pdb = app.PDBFile(args.complex_pdb)
    # modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    # modeller.addSolvent(amber_ff, numAdded=6000)
    # PDBFile.writeFile(modeller.topology, modeller.positions, open("sanitized.pdb", 'w'))
    # assert 0

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    for guest_mol in suppl:
        break

    # T = 5000
    T = 10000
    # T = 200
    dt = 0.0015
    step_sizes = np.ones(T)*dt
    cas = np.ones(T)*0.99

    num_gpus = 1
    pool = multiprocessing.Pool(num_gpus)

    all_du_dls = []

    lambda_schedule = np.linspace(0.00001, 0.99999, num=T)
    # lambda_schedule = np.linspace(0.00001, 0.01, num=T)

    epoch = 0

    init_conf = guest_mol.GetConformer(0)
    init_conf = np.array(init_conf.GetPositions(), dtype=ops.precision)
    init_conf = init_conf/10 # convert to md_units
    conf_com = com(init_conf)

    init_mol = Chem.Mol(guest_mol)

    for host_idx, host_pdb_file in enumerate([args.complex_pdb, args.solvent_pdb]):

        host_pdb = app.PDBFile(host_pdb_file)

        if host_idx == 0:
            host_name = "complex"
        elif host_idx == 1:
            host_name = "solvent"

        for mode in ['insertion', 'deletion']:
            print("Mode", mode)

            host_conf = []
            for x,y,z in host_pdb.positions:
                host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])

            host_conf = np.array(host_conf)
            init_combined_conf = np.concatenate([host_conf, init_conf])

            perm = hilbert_sort(init_combined_conf)
            # perm = np.random.permutation(np.arange(init_combined_conf.shape[0]))
            # perm = np.arange(init_combined_conf.shape[0])

            sim = simulation.Simulation(
                guest_mol,
                host_pdb,
                mode,
                step_sizes,
                cas,
                lambda_schedule,
                perm
            )

            num_conformers = 1

            guest_mol.RemoveAllConformers()
            AllChem.EmbedMultipleConfs(guest_mol, num_conformers, randomSeed=2020)

            # sample from the DG distribution
            all_args = []
            for conf_idx in range(num_conformers):
                guest_conf = guest_mol.GetConformer(conf_idx)
                guest_conf = np.array(guest_conf.GetPositions(), dtype=ops.precision)
                guest_conf = guest_conf/10 # convert to md_units
                np.random.seed(2020)
                rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
                guest_conf = np.matmul(guest_conf, rot_matrix)
                guest_conf = recenter(guest_conf, conf_com)

                x0 = np.concatenate([host_conf, guest_conf])       # combined geometry

                print("x0 shape", x0.shape)

                x0 = x0[perm]

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), init_mol)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join("frames", "epoch_"+str(epoch)+"_"+mode+"_"+host_name+"_conf_"+str(conf_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file)
                all_args.append((x0, writer,  conf_idx % num_gpus))


            results = pool.map(sim.run_forward_multi, all_args)

            sys.exit(0) 

            all_du_dls.append(results)

    all_du_dls = np.array(all_du_dls)

    error = loss.loss_dG(*all_du_dls, lambda_schedule, 250.)

    print("Error", error)

    assert 0

    # pool.close()
    # # du_dls = sim.run_forward(x0)
    # # print(du_dls)

    # assert 0

    # # parameterize the protein
    # pdb = app.PDBFile(args.protein_pdb)
    # amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')

    # system = amber_ff.createSystem(
    #     pdb.topology,
    #     nonbondedMethod=app.NoCutoff,
    #     constraints=None,
    #     rigidWater=False)

    # host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, dimension=4)
    # host_conf = []
    # for x,y,z in pdb.positions:
    #     host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    # host_conf = np.array(host_conf, dtype=np.float64)

    # # parameterize the small molecule
    # suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    # for mol in suppl:
    #     break

    # off = smirnoff.ForceField("test_forcefields/smirnoff99Frosst.offxml")
    # guest_potentials, (guest_params, guest_param_groups), guest_conf, guest_masses = forcefield.parameterize(mol, off, dimension=4)

    # print("Host Shape", host_conf.shape, "Guest Shape", guest_conf.shape)

    # combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
    #     host_potentials, guest_potentials,
    #     host_params, guest_params,
    #     host_param_groups, guest_param_groups,
    #     host_conf, guest_conf,
    #     host_masses, guest_masses)

    # x0 = combined_conf
    # v0 = np.zeros_like(x0)

    # print(x0.shape)

    # assert 0

    # gradients = []
    # for fn, fn_args in combined_potentials:
    #     gradients.append(fn(*fn_args))

    # T = 10000

    # # we have strict convergence at the end points, but for numerical reasons (1/inf)
    # # need to have this not be *exactly* closed [0, 1] but rather open (0, 1)
    # lambda_schedule = np.linspace(0.00001, 0.999999, num=T)
    # lambda_idxs = np.zeros(combined_conf.shape[0], dtype=np.int32)
    # lambda_idxs[host_conf.shape[0]:] = 1 # insertion is -1, deletion is +1

    # dt = 0.0015
    # step_sizes = np.ones(T)*dt
    # cas = np.ones(T)*0.99
    # cbs = -np.ones(combined_conf.shape[0])*0.001

    # lr = 1e-4
    # # opt_init, opt_update, get_params = optimizers.adam(lr)
    # opt_init, opt_update, get_params = optimizers.sgd(lr)

    # opt_state = opt_init(combined_params)
    # itercount = itertools.count()

    # num_epochs = 10
    # for epoch in range(num_epochs):

    #     current_params = np.asarray(get_params(opt_state))

    #     stepper = custom_ops.LambdaStepper_f64(
    #         gradients,
    #         lambda_schedule,
    #         lambda_idxs,
    #         4
    #     )

    #     ctxt = custom_ops.ReversibleContext_f64_3d(
    #         stepper,
    #         len(combined_masses),
    #         x0.reshape(-1).tolist(),
    #         v0.reshape(-1).tolist(),
    #         cas.tolist(),
    #         cbs.tolist(),
    #         step_sizes.tolist(),
    #         current_params.reshape(-1).tolist(),
    #     )

    #     start = time.time()
    #     ctxt.forward_mode()
    #     print("forward time", time.time()-start)

    #     du_dls = stepper.get_du_dl()


    #     # goal is to get all the du_dls
    #     # for x, y in zip(lambda_schedule, du_dls):
    #         # print(x, y)

    #     # plt.xlabel('lambda')
    #     # plt.ylabel('du/dl')
    #     # plt.plot(lambda_schedule, du_dls)
    #     # plt.show()


    #     work_true = 400
    #     work_pred = math_utils.trapz(du_dls, lambda_schedule)

    #     complex_insertion_work = []
    #     complex_deletion_work = []
    #     complex_dG = pymbar.bar()

    #     solvent_insertion_work = []
    #     solvent_deletion_work = []
    #     solvent_dG = pymbar.bar()

    #     final_dG = complex_dG - solvent_dG
    #     true_dG = 35

    #     loss = (true_dG - final_dG)**2

    #     # compute adjoints
    #     dL_dfinal_dG = -2*(true_dG - final_dG)
    #     dL_dcomplex_dG = dL_dfinal_dG
    #     dL_dsolvent_dG = -dL_dfinal_dG
    #     dL_dcomplex_dinsertion_work = []
    #     dL_dcomplex_ddeletion_work = []


    #     # mimic a loss comparable to bar gradients
    #     loss = np.power(work_true - work_pred, 2)/128
    #     dloss_dw = -2*(work_true - work_pred)/128

    #     print("--------epoch", epoch, "--------")
    #     print("Loss", loss, "dloss_dw", dloss_dw, "work_pred", work_pred, "work_true", work_true)
    #     print('------------')
    #     trapz_grad_fn = jax.grad(math_utils.trapz, argnums=0)

    #     # dL_d(du/dl) = dL/dw . dw/d(du/dl)
    #     dw_ddudl = trapz_grad_fn(du_dls, lambda_schedule)
    #     dloss_ddudl = dloss_dw*dw_ddudl

    #     # coords = ctxt.get_all_coords()
    #     # print(coords[0], coords[-1])
    #     # write_coords(coords, args.protein_pdb, mol, out_file)

    #     # du_dl_adjoint =  np.random.rand(du_dls.shape[0])/1000
    #     print("setting adjoints to", dloss_ddudl, "with mean", np.mean(dloss_ddudl))
    #     stepper.set_du_dl_adjoint(dloss_ddudl)

    #     # test_adjoint = np.random.rand(x0.shape[0], x0.shape[0])/10
    #     ctxt.set_x_t_adjoint(np.zeros_like(x0))
    #     start = time.time()
    #     ctxt.backward_mode()
    #     print("backward time", time.time()-start)

    #     # compute the parameter derivatives
    #     dL_dp = ctxt.get_param_adjoint_accum()

    #     # only change charges (both ligand and protein)


    #     # for p_idx, pp in enumerate(dL_dp):
    #     #     if combined_param_groups[p_idx] == 7:
    #     #         print(p_idx, pp)

    #     # assert 0


    #     param_grad = np.where(combined_param_groups == 7, dL_dp, np.zeros_like(dL_dp))
    #     print("mean_grad", np.sum(param_grad)/np.sum(combined_param_groups == 7))
    #     opt_state = opt_update(next(itercount), param_grad, opt_state)


    #     # explictly call the destructor?
    #     del ctxt
    #     del stepper
