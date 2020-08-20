import os

import numpy as np
import jax
import jax.numpy as jnp
from io import StringIO

from fe import math_utils, system
from rdkit import Chem

from training import setup_system, bootstrap
from training import service_pb2
from matplotlib import pyplot as plt
import pickle

from fe.pdb_writer import PDBWriter
from simtk.openmm.app import PDBFile

from ff.handlers import bonded, nonbonded


import time

class Timer():

    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()

    def ping(self, name):

        print(name, time.time() - self.last_time)
        self.last_time = time.time()

    def total(self):
        print("total", time.time() - start_time)

def compute_dGs(all_du_dls, lambda_schedules, du_dl_cutoff):
    stage_dGs = []
    for stage_du_dls, ti_lambdas in zip(all_du_dls, lambda_schedules):
        du_dls = []
        for lamb_full_du_dls in stage_du_dls:
            du_dls.append(jnp.mean(jnp.sum(lamb_full_du_dls[:, du_dl_cutoff:], axis=0)))
        du_dls = jnp.concatenate([du_dls])
        dG = math_utils.trapz(du_dls, ti_lambdas)
        stage_dGs.append(dG)

    return stage_dGs

def dG_TI(all_du_dls, ssc, lambda_schedules, du_dl_cutoff):
    stage_dGs = compute_dGs(all_du_dls, lambda_schedules, du_dl_cutoff)
    pred_dG = jnp.sum(stage_dGs) + ssc
    return pred_dG


def loss_fn(all_du_dls, ssc, lambda_schedules, expected_dG, du_dl_cutoff):
    """

    Parameters
    ----------
    all_du_dls: list of nd.array
        list of full_stage_du_dls (usually 3 stages)

    ssc: float
        standard state correction in kJ/mol

    lambda_schedules: list of nd.array
        combined lambda schedule

    expected_dG: float
        deltaG of unbinding. (Note the sign)

    du_dl_cutoff: int
        number of frames in the equilibration phase.
    

    """
    pred_dG = dG_TI(all_du_dls, ssc, lambda_schedules, du_dl_cutoff)
    return jnp.abs(pred_dG - expected_dG)


loss_fn_grad = jax.grad(loss_fn, argnums=(0,))


class Trainer():

    def __init__(self,
            host_pdbfile,
            stubs,
            stub_hosts,
            ff_handlers,
            lambda_schedule,
            du_dl_cutoff,
            restr_search_radius,
            restr_force_constant,
            n_frames,
            intg_steps,
            intg_dt,
            intg_temperature,
            intg_friction,
            learning_rates,
            precision):
        """
        Parameters
        ----------

        host_pdbfile: path
            location of a pdb file for the protein

        stubs: gRPC stubs
            each stub corresponds to worker address

        stub_hosts: list of str
            each item corresponds to an ip address

        ff_handlers: list of ff.handlers from handlers.nonbonded or handlers.bonded
            handlers that can parameterize the guest

        lambda_schedule: dict of (stage_idx, array)
            dictionary of stage idxs and the corresponding arrays

        du_dl_cutoff: int
            number of steps we discard when estimating <du_dl>

        restr_search_radius: float
            how far in nm when searching for restraints (typical=~0.3)

        restr_force_constant: float
            strength of the force constant used for restraints

        n_frames: int
            number of frames we store, sampled evenly from the trajectory

        intg_steps: int
            number of steps we run the simulation for

        intg_dt: float
            time step in picoseconds (typical=1.5e-3)

        intg_temperature: float
            temperature of the simulation in Kelvins (typical=300)

        intg_friction: float
            thermostat friction coefficient in 1/picseconds, (typical=40)

        learning_rates: dict of learning rates
            how much we adjust the charge by in parameter space

        precision: str
            allowed values are "single" or "double", (typical=single)

        """


        self.du_dl_cutoff = du_dl_cutoff
        self.host_pdbfile = host_pdbfile
        self.stubs = stubs
        self.stub_hosts = stub_hosts
        self.ff_handlers = ff_handlers
        self.lambda_schedule = lambda_schedule
        self.restr_search_radius = restr_search_radius
        self.restr_force_constant= restr_force_constant
        self.n_frames = n_frames
        self.intg_steps = intg_steps
        self.intg_dt = intg_dt
        self.intg_temperature = intg_temperature
        self.intg_friction = intg_friction
        self.learning_rates = learning_rates
        self.precision = precision


        futures = []
        print("resetting state on workers...")
        for stub, host in zip(self.stubs, self.stub_hosts):
            print("resetting", host)
            request = service_pb2.EmptyMessage()
            response_future = stub.ResetState.future(request)
            futures.append(response_future)

        for fut in futures:
            fut.result()


    def run_mol(self, mol, inference, run_dir, experiment_dG):
        """
        Compute the absolute unbinding free energy of given molecule. The molecule should be
        free of clashes and correctly posed within the binding pocket.

        Parameters
        ----------
        mol: Chem.ROMol
            RDKit molecule

        inference: bool
            If True, then we compute the forcefield parameter derivatives, otherwise skip.

        run_dir: str
            path of where we store all the output files

        experiment_dG: float
            experimental unbinding free energy.

        Returns
        -------
        float, float
            Predicted unbinding free energy, and loss relative to experimental dG

        """

        host_pdbfile = self.host_pdbfile
        lambda_schedule = self.lambda_schedule
        ff_handlers = self.ff_handlers
        stubs = self.stubs
        du_dl_cutoff = self.du_dl_cutoff

        host_pdb = PDBFile(host_pdbfile)
        combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdbfile, removeHs=False), mol)

        stage_forward_futures = []
        stage_state_keys = []

        stub_idx = 0

        # step 1. Prepare the jobs

        for stage, ti_lambdas in self.lambda_schedule.items():

            # print("---Starting stage", stage, '---')
            stage_dir = os.path.join(run_dir, "stage_"+str(stage))

            if not os.path.exists(stage_dir):
                os.makedirs(stage_dir)

            x0, combined_masses, ssc, final_gradients, handler_vjp_fns = setup_system.create_system(
                mol,
                host_pdb,
                ff_handlers,
                self.restr_search_radius,
                self.restr_force_constant,
                self.intg_temperature,
                stage
            )

            forward_futures = []
            state_keys = []

            for lamb_idx, lamb in enumerate(ti_lambdas):

                intg = system.Integrator(
                    steps=self.intg_steps,
                    dt=self.intg_dt,
                    temperature=self.intg_temperature,
                    friction=self.intg_friction,  
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

                # this key is used for us to chase down the forward-mode coordinates
                # when we compute derivatives in backwards mode.
                key = str(stage)+"_"+str(lamb_idx)

                request = service_pb2.ForwardRequest(
                    inference=inference,
                    system=pickle.dumps(complex_system),
                    precision=self.precision,
                    n_frames=self.n_frames,
                    key=key
                )

                stub = stubs[stub_idx % len(stubs)]
                stub_idx += 1

                # launch asynchronously
                response_future = stub.ForwardMode.future(request)
                forward_futures.append(response_future)
                state_keys.append(key)

            stage_forward_futures.append((stage, forward_futures))
            stage_state_keys.append(state_keys)

        # step 2. Run forward mode on the jobs
        all_du_dls = []
        all_lambdas = []

        for stage, stage_futures in stage_forward_futures:

            stage_dir = os.path.join(run_dir, "stage_"+str(stage))
            stage_du_dls = []

            for lamb_idx, (future, lamb) in enumerate(zip(stage_futures, lambda_schedule[stage])):

                response = future.result()

                stripped_du_dls = pickle.loads(response.du_dls)

                full_du_dls = []

                # unpack sparse du_dls into full set
                for du_dls in stripped_du_dls:
                    if du_dls is None:
                        full_du_dls.append(np.zeros(self.intg_steps, dtype=np.float64))
                    else:
                        full_du_dls.append(du_dls)

                full_du_dls = np.array(full_du_dls)
                full_energies = pickle.loads(response.energies)

                if self.n_frames > 0:
                    frames = pickle.loads(response.frames)
                    out_file = os.path.join(stage_dir, "frames_"+str(lamb_idx)+".pdb")
                    # make sure we do StringIO here as it's single-pass.
                    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                    pdb_writer = PDBWriter(combined_pdb_str, out_file)
                    pdb_writer.write_header()
                    for frame_idx, x in enumerate(frames):
                        pdb_writer.write(x*10)
                    pdb_writer.close()

                    assert full_du_dls is not None

                # we don't really want to save this full buffer
                # np.save(os.path.join(stage_dir, "lambda_"+str(lamb_idx)+"_full_du_dls"), full_du_dls)
                # total_du_dls = np.sum(full_du_dls, axis=0)

                # plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                # plt.ylabel("du_dl")
                # plt.xlabel("timestep")
                # plt.legend()
                # fpath = os.path.join(stage_dir, "lambda_du_dls_"+str(lamb_idx))
                # plt.savefig(fpath)
                # plt.clf()

                # timer.ping("d")

                # plt.plot(full_energies, label="{:.2f}".format(lamb))
                # plt.ylabel("U")
                # plt.xlabel("timestep")
                # plt.legend()

                # fpath = os.path.join(stage_dir, "lambda_energies_"+str(lamb_idx))
                # plt.savefig(fpath)
                # plt.clf()

                equil_du_dls = full_du_dls[:, du_dl_cutoff:]

                for f, du_dls in zip(final_gradients, equil_du_dls):
                    if np.any(np.abs(du_dls) > 0):
                        fname = f[0]
                        print("mol", mol.GetProp("_Name"), "stage:", stage, "lambda:", "{:.3f}".format(lamb), "\t median {:8.2f}".format(np.median(du_dls)), "\t mean", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

                total_equil_du_dls = np.sum(equil_du_dls, axis=0) # [1, T]
                print("mol", mol.GetProp("_Name"), "stage:", stage, "lambda:", "{:.3f}".format(lamb), "\t mean", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")
                stage_du_dls.append(full_du_dls)

            sum_du_dls = np.sum(stage_du_dls, axis=1) # [L,F,T], lambda windows, num forces, num frames

            ti_lambdas = lambda_schedule[stage]
            plt.boxplot(sum_du_dls[:, du_dl_cutoff:].tolist(), positions=ti_lambdas)
            plt.ylabel("du_dlambda")
            plt.savefig(os.path.join(stage_dir, "boxplot_du_dls"))
            plt.clf()

            avg_du_dls = np.mean(sum_du_dls[:, du_dl_cutoff:], axis=1)
            np.save(os.path.join(stage_dir, "avg_du_dls"), avg_du_dls)
            plt.plot(ti_lambdas, avg_du_dls)
            plt.ylabel("du_dlambda")
            plt.savefig(os.path.join(stage_dir, "avg_du_dls"))
            plt.clf()

            all_du_dls.append(stage_du_dls)
            all_lambdas.append(ti_lambdas)


        pred_dG = dG_TI(all_du_dls, ssc, all_lambdas, du_dl_cutoff)
        ci = bootstrap.ti_ci(all_du_dls, ssc, all_lambdas, du_dl_cutoff)
        loss = loss_fn(all_du_dls, ssc, all_lambdas, experiment_dG, du_dl_cutoff)

        print("mol", mol.GetProp("_Name"), "stage dGs:", compute_dGs(all_du_dls, all_lambdas, du_dl_cutoff), "ssc:", ssc)

        if not inference:

            all_adjoint_du_dls = loss_fn_grad(all_du_dls, ssc, all_lambdas, experiment_dG, du_dl_cutoff)[0]

            # step 3. run backward mode
            stage_backward_futures = []

            stub_idx = 0
            for a_idx, adjoint_du_dls in enumerate(all_adjoint_du_dls):

                futures = []
                for l_idx, adjoint_lambda_du_dls in enumerate(adjoint_du_dls):

                    key = stage_state_keys[a_idx][l_idx]

                    request = service_pb2.BackwardRequest(
                        key=key,
                        adjoint_du_dls=pickle.dumps(np.asarray(adjoint_lambda_du_dls)),
                    )
                    futures.append(stubs[stub_idx % len(stubs)].BackwardMode.future(request))
                    stub_idx += 1

                stage_backward_futures.append(futures)

            raw_charge_derivs = []
            raw_lj_derivs = []

            # (ytz): we compute the vjps using a trick:
            # since vjp_fn(y0_adjoint) + vjp_fn(y1_adjoint) == vjp_fn(y0_adjoint + y1_adjoint)
            # reduce the raw derivatives of size Q, then compute the vjp(Q_adjoint) to get P_adjoint
            for stage_futures in stage_backward_futures:
                for future in stage_futures:
                    backward_response = future.result()
                    dl_dps = pickle.loads(backward_response.dl_dps)

                    for g, dl_dp in zip(final_gradients, dl_dps):

                        # train charges only
                        if g[0] == 'Nonbonded':
                            # 0 is for charges
                            # 1 is for lj terms
                            raw_charge_derivs.append(dl_dp[0])
                            raw_lj_derivs.append(dl_dp[1])
                        elif g[0] == 'GBSA':
                            # 0 is for charges
                            # 1 is for gb terms
                            raw_charge_derivs.append(dl_dp[0])


            sum_charge_derivs = np.sum(raw_charge_derivs, axis=0)
            sum_lj_derivs = np.sum(raw_lj_derivs, axis=0)

            # (ytz): the learning rate determines the magnitude we're allowed to move each parameter.
            # every component of the derivative is adjusted so that the max element moves precisely
            # by the lr amount.
            for h, vjp_fn in handler_vjp_fns.items():

                if isinstance(h, nonbonded.SimpleChargeHandler):
                    # disable training to SimpleCharges
                    assert 0
                    h.params -= charge_gradients*self.learning_rates['charge']
                elif isinstance(h, nonbonded.AM1CCCHandler):
                    charge_gradients = vjp_fn(sum_charge_derivs)
                    if np.any(np.isnan(charge_gradients)) or np.any(np.isinf(charge_gradients)) or np.any(np.amax(np.abs(charge_gradients)) > 10000.0):
                        print("Skipping Fatal Charge Derivatives:", charge_gradients)
                    else:
                        charge_scale_factor = np.amax(np.abs(charge_gradients))/self.learning_rates['charge']
                        h.params -= charge_gradients/charge_scale_factor
                elif isinstance(h, nonbonded.LennardJonesHandler):
                    lj_gradients = vjp_fn(sum_lj_derivs)
                    if np.any(np.isnan(lj_gradients)) or np.any(np.isinf(lj_gradients)) or np.any(np.amax(np.abs(lj_gradients)) > 10000.0):
                        print("Skipping Fatal LJ Derivatives:", lj_gradients)
                    else:
                        lj_sig_scale = np.amax(np.abs(lj_gradients[:, 0]))/self.learning_rates['lj'][0]
                        lj_eps_scale = np.amax(np.abs(lj_gradients[:, 1]))/self.learning_rates['lj'][1]
                        lj_scale_factor = np.array([lj_sig_scale, lj_eps_scale])
                        h.params -= lj_gradients/lj_scale_factor

        return pred_dG, ci, loss

