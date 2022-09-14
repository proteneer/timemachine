import numpy as np
from timemachine import potentials
from timemachine.potentials import nonbonded
import functools

def prepare_lj_system(
    x,
    E,  # number of exclusions
    lambda_plane_idxs,
    lambda_offset_idxs,
    p_scale,
    tip3p,
    cutoff=100.0,
    precision=np.float64,
):

    assert x.ndim == 2
    N = x.shape[0]
    # D = x.shape[1]

    sig_params = np.random.rand(N) / p_scale
    eps_params = np.random.rand(N)
    lj_params = np.stack([sig_params, eps_params], axis=1)

    if tip3p:
        mask = []
        for i in range(N):
            if i % 3 == 0:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask)
        eps_params = lj_params[:, 1]
        tip_params = np.where(mask, eps_params, 0)
        lj_params[:, 1] = tip_params

    atom_idxs = np.arange(N)
    exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

    lj_scales = np.random.rand(E)

    test_potential = potentials.LennardJones(
        exclusion_idxs, lj_scales, lambda_plane_idxs, lambda_offset_idxs, cutoff, precision=precision
    )

    ref_potential = functools.partial(
        nonbonded.lennard_jones_v2,
        exclusion_idxs=exclusion_idxs,
        lj_scales=lj_scales,
        cutoff=cutoff,
        lambda_plane_idxs=lambda_plane_idxs,
        lambda_offset_idxs=lambda_offset_idxs,
    )

    return lj_params, ref_potential, test_potential


# def prepare_es_system(
#     x,
#     E, # number of exclusions
#     lambda_offset_idxs,
#     p_scale,
#     cutoff,
#     precision=np.float64):

#     N = x.shape[0]
#     D = x.shape[1]

#     charge_params = (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(ONE_4PI_EPS0)

#     atom_idxs = np.arange(N)
#     exclusion_idxs = np.random.choice(atom_idxs, size=(E, 2), replace=False)
#     exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32).reshape(-1, 2)

#     charge_scales = np.random.rand(E)

#     # beta = np.random.rand()*2

#     beta = 2.0

#     test_potential = potentials.Electrostatics(
#         exclusion_idxs,
#         charge_scales,
#         lambda_offset_idxs,
#         beta,
#         cutoff,
#         precision=precision
#     )

#     ref_total_energy = functools.partial(
#         nonbonded.electrostatics_v2,
#         exclusion_idxs=exclusion_idxs,
#         charge_scales=charge_scales,
#         beta=beta,
#         cutoff=cutoff,
#         lambda_offset_idxs=lambda_offset_idxs
#     )

#     return charge_params, ref_total_energy, test_potential

