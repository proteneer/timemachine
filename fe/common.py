import time
import numpy as np

from system import forcefield
import sys

import jax

from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from system import custom_functionals

from fe.utils import to_md_units, write

def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18

def dU_dlambda(dE_dx, nha):
    """
    Compute the 4 dimensional dU/dlambda via decoupling.
    """
    # print("dE_dx shape", dE_dx.shape, "nha", nha, "last", dE_dx[nha:, 3:], dE_dx[nha:, :3])
    return np.sum(dE_dx[nha:, 3:])

def ns_per_day(delta, timestep_in_fs):
    return (timestep_in_fs/delta)*(86400)*1e-6

def print_displacement(starting, current):
    disp = np.linalg.norm(starting-current, axis=-1)
    print("mean/max displacement", np.mean(disp), np.max(disp))
    centered_starting = starting - np.mean(starting, axis=0, keepdims=True)
    centered_current = current - np.mean(current, axis=0, keepdims=True)
    centered_disp = np.linalg.norm(centered_starting-centered_current, axis=-1)
    print("mean/max centered displacement", np.mean(centered_disp), np.max(centered_disp))

# one-step insertion
def noneq_switching(ctxt,
    opt,
    coeff_e,
    inference,
    nha,
    writer_fn,
    mol_name,
    coeff_a=0.95,
    max_iter=10000):

    opt.set_coeff_a(coeff_a)
    opt.set_coeff_e(coeff_e) # how much lambda is multiplied by

    dx_xis = []
    dx_dps = []
    lambdas = []
    du_dls = []
    d2u_dldps = []
    bufs = []

    start_time = time.time()

    frames = 0
    for i in range(max_iter):

        collect = i % 100 == 0 or i == max_iter-1

        # (ytz): Due to the staggered nature of the integrator, we collect geometries before calling step() and
        # collect forces and other derivatives after calling step(). Otherwise the computed values will be inconsistent.

        if collect:
            xi = ctxt.get_x()     
            lamb = np.mean(xi[nha:, 3:])
            lambdas.append(lamb)

        ctxt.step()

        if collect:
            E = ctxt.get_E()

            dx = ctxt.get_dE_dx()
            if not inference:
                dx_dp = ctxt.get_dx_dp()
            else:
                dx_dp = 0

            du_dl = dU_dlambda(ctxt.get_dE_dx(), nha)
            du_dls.append(du_dl)

            time_per_leg = (time.time() - start_time)/(i+1) * (max_iter/3600)

            print("ran", i+1, "steps in", time.time() - start_time)
            speed = ns_per_day(time.time() - start_time, i)
            print("step", i, "lambda", lamb, "dt", opt.get_dt(), "Energy", E, "du/dl", du_dl, "speed", speed, "ns/day",  "TPL:", time_per_leg, "hrs", mol_name, "3d max force component", np.amax(np.abs(dx[:, :3])), "3d norm force", np.linalg.norm(dx[:, :3]), "dxdp max/min", np.amax(dx_dp), np.amin(dx_dp), end=' ')

            if np.isnan(E):
                print("warning, nans found")
                assert 0
            else:
                if i % 200 == 0 or i == max_iter-1:
                    writer_fn(xi*10)
                    frames += 1

        # (ytz): this needs to be called *after* in order for the derivatives to correct. the sgemm buffers at step i correspond
        # to the state at the *previous* time step.
        if collect and not inference:
            sgemm_sum = ctxt.get_sgemm_sum()

            d2u_dldp = np.sum(sgemm_sum[:, nha:, 3:], axis=(1,2))
            d2u_dldps.append(d2u_dldp)
            print("max/min d2u_dldp", np.amax(d2u_dldp), np.amin(d2u_dldp))
        elif collect:
            print(" ")

    del ctxt

    lambdas = np.array(lambdas)
    du_dls = np.array(du_dls)

    # np.savez(mol_name, lambdas, du_dls)

    # sort them just in case lambda isn't linear
    # perm = np.argsort(lambdas)
    # lambdas = lambdas[perm]
    # du_dls = du_dls[perm]

    work = np.trapz(du_dls, lambdas)

    return work

def minimize(
    mol_name,
    num_host_atoms,
    potentials,
    params,
    conf,
    masses,
    dp_idxs, # if this is an empty list then we're doing inference
    insertion=False,
    writer_fn=None):
    """
    Minimize a structure.
    """
    frames = 0 

    if len(dp_idxs) == 0:
        inference = True
    else:
        inference = False

    num_atoms = len(masses)
    potentials = forcefield.merge_potentials(potentials)

    dt = 0.001
    ca, cb, cc = langevin_coefficients(
        temperature=300,
        dt=dt,
        friction=100, # (ytz) probably need to double this?
        masses=np.ones_like(masses),
    )

    print(ca, cb, cc)

    m_dt, m_ca, m_cb, m_cc = dt, 0.0, np.ones_like(cb)/10000, np.zeros_like(masses)

    num_ligand_atoms = conf.shape[0] - num_host_atoms
    offset = (conf.shape[0]*3) + num_ligand_atoms
    # offset = conf.shape[0]*4
    # offset = conf.shape[0]*conf.shape[1]
    print("OFFSET SGEMM", offset, "vs", conf.shape[0]*4, offset/(conf.shape[0]*4))

    opt = custom_functionals.langevin_optimizer(
        dt,
        4,
        m_ca,
        m_cb.astype(custom_functionals.precision),
        m_cc.astype(custom_functionals.precision),
        offset
    )

    dp_idxs = np.array(dp_idxs).astype(np.int32)

    num_atoms = conf.shape[0]
    num_dimensions = 4
    
    lamb_start = 5.0

    d4_t = np.zeros((num_atoms, num_dimensions), dtype=custom_functionals.precision)
    d4_t_lambdas = np.zeros((num_atoms, num_dimensions), dtype=custom_functionals.precision) + lamb_start

    # set coordinates
    d4_t[:num_host_atoms, :3] = conf[:num_host_atoms, :3]
    d4_t[num_host_atoms:, :3] = conf[num_host_atoms:, :3]
    d4_t[num_host_atoms:, 3:] = d4_t_lambdas[num_host_atoms:, 3:]

    x_t = d4_t
    v_t = np.zeros_like(x_t)

    if inference:
        ctxt = custom_functionals.inference_context(
            potentials,
            opt,
            params.astype(custom_functionals.precision),
            x_t.astype(custom_functionals.precision),
            v_t.astype(custom_functionals.precision)
        )
    else:
        ctxt = custom_functionals.context(
            potentials,
            opt,
            params.astype(custom_functionals.precision),
            x_t.astype(custom_functionals.precision),
            v_t.astype(custom_functionals.precision),
            dp_idxs.astype(np.int32)
        )

    start_time = time.time()

    lamb_final = 1e-4 # final lambda desired

    max_iter = 10000
    coeff_insertion = np.power(lamb_final/lamb_start, 1/max_iter)
    coeff_deletion = np.power(1/np.power(coeff_insertion, max_iter), 1/max_iter)
    print("CD", coeff_deletion)

    middle = lamb_start*np.power(coeff_insertion, max_iter)
    end = middle*np.power(coeff_deletion, max_iter)
    print("schedule", lamb_start, "->", middle, "->", end)
    # print("insertion, lambda", lambda_start*np.power(coeff_insertion, max_iter))
    # print("deletion, lambda", lambda_start*np.power(coeff_insertion, max_iter)*np.power(coeff_deletion, max_iter))

    # minimize the geometries
    dt = 1e-4
    dt_increment = 1.005
    max_dt = 0.013

    # debug
    # for i in range(1000):    
    st = time.time()
    Es = []
    for i in range(100):
        dt *= dt_increment
        dt = min(dt, max_dt)
        opt.set_dt(dt)
        ctxt.step()

        if i % 100 == 0:
            E = ctxt.get_E()
            print(dt, i, E)
            Es.append(E)
            xi = ctxt.get_x()
            # if np.abs(E) < 1e6:
                # writer_fn(xi*10)
            if len(Es) > 2 and np.abs(Es[-1] - Es[-2]) > 1e3:
                raise Exception("FUCK")
            else:
                writer_fn(xi*10)


    time_per_leg = (time.time() - start_time)/(i+1) * (max_iter/3600)

    print("time per leg",  time_per_leg)

    sys.exit(0)

    # debug
    hess = ctxt.get_d2E_dx2()
    print("hess size", hess.size)
    print("hess num zeros", np.sum(hess == 0), "sparsity:", np.sum(hess == 0)/hess.size)

    dxdp = ctxt.get_dx_dp()
    print("dxdp size", dxdp.size, "dxdp shape", dxdp.shape)
    print("dxdp num zeros", np.sum(dxdp == 0), "sparsity:", np.sum(dxdp == 0)/dxdp.size)

    np.savez("hvp_test", hess=hess, dxdp=dxdp)

    assert 0

    time_per_leg = (time.time() - start_time)/(i+1) * (max_iter/3600)

    print("time per leg",  time_per_leg)


    # assert 0
    print("time taken to minimize:", time.time()-st)
    sys.exit(0)

    print("start")



    coeff_a = 0.0

    dG_insertion = noneq_switching(ctxt, opt, inference=inference, coeff_e=coeff_insertion, nha=num_host_atoms, mol_name=mol_name+"_insertion", writer_fn=writer_fn, coeff_a=coeff_a, max_iter=max_iter)
    
    assert 0

    dG_deletion = noneq_switching(ctxt, opt, inference=inference, coeff_e=coeff_deletion, nha=num_host_atoms, mol_name=mol_name+"_deletion", writer_fn=writer_fn, coeff_a=coeff_a, max_iter=max_iter)

    print("dG_insertion", dG_insertion)
    print("dG_deletion", dG_deletion)

    return dG_insertion, dG_deletion
    # assert 0

    # opt.set_coeff_a(0.95)
    # opt.set_coeff_e(lambda_decrement) # how much lambda is multiplied by

    # starting = x_t

    # print("---stage 1 insertion")

    # dx_xis = []
    # dx_dps = []
    # lambdas = []
    # du_dls = []
    # d2u_dldps = []
    # bufs = []

    # for i in range(insertion_max_iter):
    #     dt *= dt_increment
    #     dt = min(dt, max_dt)
    #     opt.set_dt(dt)

    #     collect = i % 100 == 0 or i == insertion_max_iter-1

    #     if collect:
    #         E = ctxt.get_E()
    #         xi = ctxt.get_x()

    #         dx = ctxt.get_dE_dx()
    #         if not inference:
    #             dx_dp = ctxt.get_dx_dp()
    #         else:
    #             dx_dp = 0

    #         # (ytz) this is a slight bug, should *always* be taken after the step to avoid bug otherwise there's a gradient error, since the gradients are taken *after* the fact.
    #         # ie. at step = 0 this is ZERO, which is incorrect.
    #         du_dl = dU_dlambda(ctxt.get_dE_dx(), num_host_atoms)
    #         du_dls.append(du_dl)

    #         lamb = np.mean(xi[num_host_atoms:, 3:])
    #         lambdas.append(lamb)
    #         speed = ns_per_day(time.time() - start_time, i)

    #         print("step", i, "lambda", lamb, "dt", opt.get_dt(), "Energy", E, "du/dl", du_dl, "speed", speed, "ns/day", mol_name, "3d max force component", np.amax(np.abs(dx[:, :3])), "3d norm force", np.linalg.norm(dx[:, :3]), "dxdp max/min", np.amax(dx_dp), np.amin(dx_dp), end=' ')

    #         if np.isnan(E):
    #             print("warning, nans found")
    #             assert 0
    #         else:
    #             if i % 200 == 0 or i == insertion_max_iter-1:
    #                 writer_fn(xi*10, frames)
    #                 frames += 1

    #         all_xis.append(xi)

    #     ctxt.step()

    #     # (ytz): this needs to be called *after* in order for the derivatives to correct. the sgemm buffers at step i correspond
    #     # to the state at the *previous* time step.
    #     if collect and not inference:
    #         sgemm_sum = ctxt.get_sgemm_sum()

    #         d2u_dldp = np.sum(sgemm_sum[:, num_host_atoms:, 3:], axis=(1,2))
    #         d2u_dldps.append(d2u_dldp)
    #         print("max/min d2u_dldp", np.amax(d2u_dldp), np.amin(d2u_dldp))
    #     elif collect:
    #         print(" ")

    # for b_idx, b in enumerate(bufs):
    #     writer_fn(b, b_idx)

    # del ctxt

    # lambdas = np.array(lambdas)
    # du_dls = np.array(du_dls)

    # np.savez(mol_name, lambdas, du_dls)

    # # sort them just in case lambda isn't linear
    # perm = np.argsort(lambdas)
    # lambdas = lambdas[perm]
    # du_dls = du_dls[perm]

    # dG = np.trapz(du_dls, lambdas)

    # if not inference:
    #     d2u_dldps = np.array(d2u_dldps) # [num_lambdas, P]
    #     d2u_dldps = d2u_dldps[perm]
    #     num_params = d2u_dldps.shape[-1]
    #     assert len(d2u_dldps.shape) == 2
    #     dG_grads = []
    #     for p_idx in range(num_params):
    #         dG_grads.append(np.trapz(d2u_dldps[:, p_idx], lambdas))
    #     dG_grads = np.array(dG_grads)
    # else:
    #     dG_grads = None

    # print("dG", dG)

    # return dG, dG_grads, all_xis



# def batch_compute_d2u_dldp(energies, params, all_xs, all_dx_dps, dp_idxs, num_host_atoms):

#     for xs, dx_dps in zip(all_xs, all_dx_dps):
#         assert len(xs.shape) == 2 # [N, 3]
#         assert len(dx_dps.shape) == 3 # [P, N, 3]


#     all_xs = np.array(all_xs) # convert to numpy
#     all_dx_dps = np.array(all_dx_dps) # convert to numpy

#     start_time = time.time()

#     mixed_partials = []
#     hessians = []

#     print("Computing energies")
#     for p in energies:
#         print(p)
#         _, _, ph, _, pmp  = p.derivatives(all_xs, params, dp_idxs)
#         mixed_partials.append(pmp)
#         hessians.append(ph)

#     hessians = np.sum(hessians, axis=0)[0]
#     mixed_part = np.sum(mixed_partials, axis=0)[0]

#     print("Computing tensor contractions")

#     hess_idxs = jax.ops.index[:, num_host_atoms:, 3:, :, :3]
#     mp_idxs = jax.ops.index[:, num_host_atoms:, 3:]
#     dx_dp_idxs = jax.ops.index[:, :, :, :3]

#     lhs = np.einsum('bijkl,bmkl->bmij', hessians[hess_idxs], dx_dps[dx_dp_idxs]) # correct only up to main hessian
#     rhs = mixed_part[mp_idxs]

#     # lhs + rhs has shape [P, num_atoms-num_host_atoms, 1] 
#     all_d2u_dldp = np.sum(lhs+rhs, axis=(2,3)) # B P N 4 -> P B
    

#     # all_d2u_dldps = []
#     # all_d2u_dldps.append(d2u_dldp)
#     print("Time taken to compute derivatives:", time.time()-start_time)

#     return all_d2u_dldps