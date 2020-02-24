import numpy as np
import simtk.unit

def set_velocities_to_temperature(n_atoms, temperature, masses):
    assert 0 # don't call this yet
    v_t = np.random.normal(size=(n_atoms, 3))
    velocity_scale = np.sqrt(constants.BOLTZ*temperature/np.expand_dims(masses, -1))
    return v_t*velocity_scale

def compute_d2e_dxdp(energies, params, xs, dp_idxs):
    mixed_partials = []
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)

    mixed_part = np.sum(mixed_partials, axis=0)[0]
    return mixed_part

def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


def write(xyz, masses, recenter=True):
    if recenter:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + '\n'
    buf += 'timemachine\n'
    for m, (x,y,z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = 'C'
        elif int(round(m)) == 14:
            symbol = 'N'
        elif int(round(m)) == 16:
            symbol = 'O'
        elif int(round(m)) == 32:
            symbol = 'S'
        elif int(round(m)) == 35:
            symbol = 'Cl'
        elif int(round(m)) == 1:
            symbol = 'H'
        elif int(round(m)) == 31:
            symbol = 'P'
        elif int(round(m)) == 19:
            symbol = 'F'
        elif int(round(m)) == 80:
            symbol = 'Br'
        elif int(round(m)) == 127:
            symbol = 'I'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf


# unused
def compute_d2u_dldp(energies, params, xs, dx_dps, dp_idxs, num_host_atoms):

    all_d2u_dldps = []
    start_time = time.time()
    # for xs, dx_dps in zip(all_xs, all_dx_dps):

    assert len(xs.shape) == 2 # [N, 3]
    assert len(dx_dps.shape) == 3 # [P, N, 3]

    mixed_partials = []
    hessians = []
    # we need to compute this separately since the context's sgemm call overwrites
    # the values of d2u_dxdp
    # batched call, redo this in batch mode? (all hessians will be quite large though)

    # fuck this is so slow
    print("Computing energies")
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)
        hessians.append(ph)
    
    print("Computing tensor contractions")
    hessians = np.sum(hessians, axis=0)[0]
    mixed_part = np.sum(mixed_partials, axis=0)[0]

    hess_idxs = jax.ops.index[num_host_atoms:, 3:, :, :3]
    dx_dp_idxs = jax.ops.index[:, :, :3]
    mp_idxs = jax.ops.index[:, num_host_atoms:, 3:]
    lhs = np.einsum('ijkl,mkl->mij', hessians[hess_idxs], dx_dps[dx_dp_idxs]) # correct only up to main hessian
    rhs = mixed_part[mp_idxs]

    # lhs + rhs has shape [P, num_atoms-num_host_atoms, 1] 
    d2u_dldp = np.sum(lhs+rhs, axis=(1,2)) # P N 4 -> P

    print("Done")

    all_d2u_dldps.append(d2u_dldp)    
    print("Time taken to compute derivatives:", time.time()-start_time)

    return all_d2u_dldps
