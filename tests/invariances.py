from jax.test_util import check_grads


def assert_potential_invariance_aperiodic(energy_fn, x0, params):

    # check without box
    check_grads(energy_fn, (x0, params, None), order=1, eps=1e-5)
    check_grads(energy_fn, (x0, params, None), order=2, eps=1e-7)

    # (ytz: TODO) test translational and rotational invariance of
    # energy and its derivatives, with and without box
    

def assert_potential_invariance(energy_fn, x0, params, box):

    # check without box
    check_grads(energy_fn, (x0, params, None), order=1, eps=1e-5)
    check_grads(energy_fn, (x0, params, None), order=2, eps=1e-7)

    # check with box
    check_grads(energy_fn, (x0, params, box), order=1, eps=1e-5)
    check_grads(energy_fn, (x0, params, box), order=2, eps=1e-7)

    # (ytz: TODO) test translational and rotational invariance of
    # energy and its derivatives, with and without box
    