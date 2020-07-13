import os

import numpy as np
import jax
import jax.numpy as jnp

from fe import math_utils, system
from rdkit import Chem

from training import setup_system
from training import service_pb2
from matplotlib import pyplot as plt
import pickle

from ff.handlers import bonded, nonbonded


def dG_TI(all_du_dls, lambda_schedules, du_dl_cutoff):
    stage_dGs = []
    for stage_du_dls, ti_lambdas in zip(all_du_dls, lambda_schedules):
        du_dls = []
        for lamb_full_du_dls in stage_du_dls:
            du_dls.append(jnp.mean(jnp.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0)))
        du_dls = jnp.concatenate([du_dls])
        dG = math_utils.trapz(du_dls, ti_lambdas)
        stage_dGs.append(dG)

    pred_dG = jnp.sum(stage_dGs)
    return pred_dG


def loss_fn(all_du_dls, lambda_schedules, expected_dG, du_dl_cutoff):
    """

    Parameters
    ----------
    all_du_dls: list of nd.array
        list of full_stage_du_dls (usually 3 stages)

    lambda_schedules: list of nd.array
        combined lambda schedule

    expected_dG: float
        deltaG of unbinding. (Note the sign)

    du_dl_cutoff: int
        number of frames in the equilibration phase.
    

    """
    pred_dG = dG_TI(all_du_dls, lambda_schedules, du_dl_cutoff)
    return jnp.abs(pred_dG - expected_dG)


loss_fn_grad = jax.grad(loss_fn, argnums=(0,))


class Trainer():

    def __init__(self,
            host_pdb,
            stubs,
            ff_handlers,
            lambda_schedule,
            core_smarts,
            restr_force,
            restr_alpha,
            restr_count,
            steps,
            precision):

        n_workers = len(stubs)
        n_lambdas = np.sum([len(x) for x in lambda_schedule])
        assert n_workers == n_lambdas

        self.host_pdb = host_pdb
        self.stubs = stubs
        self.ff_handlers = ff_handlers
        self.lambda_schedule = lambda_schedule
        self.core_smarts = core_smarts
        self.restr_force = restr_force
        self.restr_alpha = restr_alpha
        self.restr_count = restr_count
        self.steps = steps
        self.precision = precision

        print("resetting state on workers...")

        futures = []
        for stub in self.stubs:
            request = service_pb2.EmptyMessage()
            response_future = stub.ResetState.future(request)
            futures.append(response_future)

        for fut in futures:
            fut.result()
            

    def run_mol(self, mol, inference, run_dir, experiment_dG):

        host_pdb = self.host_pdb
        lambda_schedule = self.lambda_schedule
        ff_handlers = self.ff_handlers
        stubs = self.stubs

        core_query = Chem.MolFromSmarts(self.core_smarts)
        core_atoms = mol.GetSubstructMatch(core_query)

        # stage 1 ti_lambdas
        stage_forward_futures = []
        stub_idx = 0

        # step 1. Prepare the jobs
        for stage in [0,1,2]:

            # print("---Starting stage", stage, '---')
            stage_dir = os.path.join(run_dir, "stage_"+str(stage))

            if not os.path.exists(stage_dir):
                os.makedirs(stage_dir)

            x0, combined_masses, final_gradients, final_vjp_fns = setup_system.create_system(
                mol,
                host_pdb,
                ff_handlers,
                stage,
                core_atoms,
                self.restr_force,
                self.restr_alpha,
                self.restr_count
            )

            ti_lambdas = lambda_schedule[stage]

            forward_futures = []

            for lamb_idx, lamb in enumerate(ti_lambdas):

                intg = system.Integrator(
                    steps=self.steps,
                    dt=1.5e-3,
                    temperature=300.0,
                    friction=40.0,  
                    masses=combined_masses,
                    lamb=lamb,
                    seed=np.random.randint(np.iinfo(np.int32).max)
                )

                complex_system = system.System(
                    x0,
                    np.zeros_like(x0),
                    final_gradients,
                    intg
                )

                request = service_pb2.ForwardRequest(
                    inference=inference,
                    system=pickle.dumps(complex_system),
                    precision=self.precision
                )

                stub = stubs[stub_idx]
                stub_idx += 1

                # launch asynchronously
                response_future = stub.ForwardMode.future(request)
                forward_futures.append(response_future)

            stage_forward_futures.append(forward_futures)

        # step 2. Run forward mode on the jobs

        du_dl_cutoff = 4000

        all_du_dls = []
        for stage_idx, stage_futures in enumerate(stage_forward_futures):
            stage_du_dls = []
            for future in stage_futures:

                response = future.result()

                full_du_dls = pickle.loads(response.du_dls)
                full_energies = pickle.loads(response.energies)

                assert full_du_dls is not None

                np.save(os.path.join(stage_dir, "lambda_"+str(lamb_idx)+"_full_du_dls"), full_du_dls)
                total_du_dls = np.sum(full_du_dls, axis=0)

                plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                plt.ylabel("du_dl")
                plt.xlabel("timestep")
                plt.legend()
                fpath = os.path.join(stage_dir, "lambda_du_dls_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                plt.plot(full_energies, label="{:.2f}".format(lamb))
                plt.ylabel("U")
                plt.xlabel("timestep")
                plt.legend()

                fpath = os.path.join(stage_dir, "lambda_energies_"+str(lamb_idx))
                plt.savefig(fpath)
                plt.clf()

                equil_du_dls = full_du_dls[:, du_dl_cutoff:]

                for f, du_dls in zip(final_gradients, equil_du_dls):
                    fname = f[0]
                    print("lambda:", "{:.3f}".format(lamb), "\t median {:8.2f}".format(np.median(du_dls)), "\t mean", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

                total_equil_du_dls = np.sum(equil_du_dls, axis=0) # [1, T]
                print("lambda:", "{:.3f}".format(lamb), "\t mean", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")

                stage_du_dls.append(full_du_dls)

            all_du_dls.append(stage_du_dls)

        pred_dG = dG_TI(all_du_dls, lambda_schedule, du_dl_cutoff)
        loss = loss_fn(all_du_dls, lambda_schedule, experiment_dG, du_dl_cutoff)

        if not inference:

            all_adjoint_du_dls = loss_fn_grad(all_du_dls, lambda_schedule, experiment_dG, du_dl_cutoff)[0]

            # step 3. run backward mode
            stage_backward_futures = []

            stub_idx = 0
            for stage_idx, adjoint_du_dls in enumerate(all_adjoint_du_dls):

                futures = []
                for lambda_du_dls in adjoint_du_dls:
                    request = service_pb2.BackwardRequest(
                        adjoint_du_dls=pickle.dumps(np.asarray(lambda_du_dls)),
                    )
                    futures.append(stubs[stub_idx].BackwardMode.future(request))
                    stub_idx += 1

                stage_backward_futures.append(futures)

            charge_derivatives = []
            gb_derivatives = []

            for stage_idx, stage_futures in enumerate(stage_backward_futures):
                for future in stage_futures:
                    backward_response = future.result()
                    dl_dps = pickle.loads(backward_response.dl_dps)

                    for g, vjp_fn, dl_dp in zip(final_gradients, final_vjp_fns, dl_dps):

                        # train charges only
                        if g[0] == 'Nonbonded':
                            # 0 is for charges
                            # 1 is for lj terms
                            charge_derivatives.append(vjp_fn[0](dl_dp[0]))
                        elif g[0] == 'GBSA':
                            # 0 is for charges
                            # 1 is for gb terms
                            charge_derivatives.append(vjp_fn[0](dl_dp[0]))
                            gb_derivatives.append(vjp_fn[1](dl_dp[1]))


            charge_gradients = np.sum(charge_derivatives, axis=0) # reduce
            charge_lr = 1e-3

            for h in ff_handlers:
                if isinstance(h, nonbonded.SimpleChargeHandler):
                    h.params -= charge_gradients*charge_lr

        return pred_dG, loss
