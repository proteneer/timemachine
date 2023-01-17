import functools

import numpy as np


def get_batch_U_fns(bps):
    # return a function that takes in coords, boxes
    all_U_fns = []
    for bp in bps:

        def batch_U_fn(xs, boxes, bp_impl):
            Us = []
            for x, box in zip(xs, boxes):
                # tbd optimize to "selective" later
                _, U = bp_impl.execute(x, box)
                Us.append(U)
            return np.array(Us)

        # extra functools.partial is needed to deal with closure jank
        all_U_fns.append(functools.partial(batch_U_fn, bp_impl=bp))

    return all_U_fns
