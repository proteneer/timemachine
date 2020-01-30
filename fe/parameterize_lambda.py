import argparse
import time
import numpy as np
from io import StringIO
import itertools

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

import jax
import rdkit
from rdkit import Chem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from fe.utils import to_md_units, write
from fe import math_utils
from system import serialize, forcefield

from openforcefield.typing.engines import smirnoff
from matplotlib import pyplot as plt

from jax.experimental import optimizers


def write_coords(frames, pdb_path, romol, outfile, num_frames=100):
    combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(pdb_path, removeHs=False), mol)
    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
    cpdb = app.PDBFile(combined_pdb_str)
    PDBFile.writeHeader(cpdb.topology, outfile)

    interval = max(1, frames.shape[0]//num_frames)

    for frame_idx, x in enumerate(frames):
        if frame_idx % interval == 0:
            PDBFile.writeModel(cpdb.topology, x*10, outfile, frame_idx)

    PDBFile.writeFooter(pdb.topology, outfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--out_pdb', type=str)
    parser.add_argument('--protein_pdb', type=str)
    parser.add_argument('--ligand_sdf', type=str)
    args = parser.parse_args()

    # parameterize the protein
    pdb = app.PDBFile(args.protein_pdb)
    amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')
    system = amber_ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)

    host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, dimension=4)
    host_conf = []
    for x,y,z in pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf, dtype=np.float64)


    # parameterize the small molecule
    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    for mol in suppl:
        break

    off = smirnoff.ForceField("test_forcefields/smirnoff99Frosst.offxml")
    guest_potentials, (guest_params, guest_param_groups), guest_conf, guest_masses = forcefield.parameterize(mol, off, dimension=4)

    print("Host Shape", host_conf.shape, "Guest Shape", guest_conf.shape)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials,
        host_params, guest_params,
        host_param_groups, guest_param_groups,
        host_conf, guest_conf,
        host_masses, guest_masses)

    x0 = combined_conf
    v0 = np.zeros_like(x0)

    gradients = []
    for fn, fn_args in combined_potentials:
        gradients.append(fn(*fn_args))

    T = 10000

    # we have strict convergence at the end points, but for numerical reasons (1/inf)
    # need to have this not be *exactly* closed [0, 1] but rather open (0, 1)
    lambda_schedule = np.linspace(0.00001, 0.999999, num=T)
    lambda_idxs = np.zeros(combined_conf.shape[0], dtype=np.int32)
    lambda_idxs[host_conf.shape[0]:] = 1 # insertion is -1, deletion is +1

    dt = 0.0015
    step_sizes = np.ones(T)*dt
    cas = np.ones(T)*0.99
    cbs = -np.ones(combined_conf.shape[0])*0.001

    lr = 1e-4
    # opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(combined_params)
    itercount = itertools.count()

    num_epochs = 10
    for epoch in range(num_epochs):

        num_gpus = 8

        batch_size = num_gpus



        current_params = np.asarray(get_params(opt_state))

        stepper = custom_ops.LambdaStepper_f64(
            gradients,
            lambda_schedule,
            lambda_idxs,
            1
        )

        ctxt = custom_ops.ReversibleContext_f64_3d(
            stepper,
            len(combined_masses),
            x0.reshape(-1).tolist(),
            v0.reshape(-1).tolist(),
            cas.tolist(),
            cbs.tolist(),
            step_sizes.tolist(),
            current_params.reshape(-1).tolist(),
        )

        start = time.time()
        ctxt.forward_mode()
        print("forward time", time.time()-start)

        du_dls = stepper.get_du_dl()

        # for x, y in zip(lambda_schedule, du_dls):
            # print(x, y)

        # plt.xlabel('lambda')
        # plt.ylabel('du/dl')
        # plt.plot(lambda_schedule, du_dls)
        # plt.show()


        work_true = 400
        work_pred = math_utils.trapz(du_dls, lambda_schedule)

        complex_insertion_work = []
        complex_deletion_work = []
        complex_dG = pymbar.bar()

        solvent_insertion_work = []
        solvent_deletion_work = []
        solvent_dG = pymbar.bar()

        final_dG = complex_dG - solvent_dG
        true_dG = 35

        loss = (true_dG - final_dG)**2

        # compute adjoints
        dL_dfinal_dG = -2*(true_dG - final_dG)
        dL_dcomplex_dG = dL_dfinal_dG
        dL_dsolvent_dG = -dL_dfinal_dG
        dL_dcomplex_dinsertion_work = []
        dL_dcomplex_ddeletion_work = []


        # mimic a loss comparable to bar gradients
        loss = np.power(work_true - work_pred, 2)/128
        dloss_dw = -2*(work_true - work_pred)/128

        print("--------epoch", epoch, "--------")
        print("Loss", loss, "dloss_dw", dloss_dw, "work_pred", work_pred, "work_true", work_true)
        print('------------')
        trapz_grad_fn = jax.grad(math_utils.trapz, argnums=0)

        # dL_d(du/dl) = dL/dw . dw/d(du/dl)
        dw_ddudl = trapz_grad_fn(du_dls, lambda_schedule)
        dloss_ddudl = dloss_dw*dw_ddudl

        # coords = ctxt.get_all_coords()
        # print(coords[0], coords[-1])
        # write_coords(coords, args.protein_pdb, mol, out_file)

        # du_dl_adjoint =  np.random.rand(du_dls.shape[0])/1000
        print("setting adjoints to", dloss_ddudl, "with mean", np.mean(dloss_ddudl))
        stepper.set_du_dl_adjoint(dloss_ddudl)

        # test_adjoint = np.random.rand(x0.shape[0], x0.shape[0])/10
        ctxt.set_x_t_adjoint(np.zeros_like(x0))
        start = time.time()
        ctxt.backward_mode()
        print("backward time", time.time()-start)

        # compute the parameter derivatives
        dL_dp = ctxt.get_param_adjoint_accum()

        # only change charges (both ligand and protein)


        # for p_idx, pp in enumerate(dL_dp):
        #     if combined_param_groups[p_idx] == 7:
        #         print(p_idx, pp)

        # assert 0


        param_grad = np.where(combined_param_groups == 7, dL_dp, np.zeros_like(dL_dp))
        print("mean_grad", np.sum(param_grad)/np.sum(combined_param_groups == 7))
        opt_state = opt_update(next(itercount), param_grad, opt_state)


        # explictly call the destructor?
        del ctxt
        del stepper