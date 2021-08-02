from jax import tree_util, numpy as np
import numpy as onp
from typing import Tuple, Callable


def flatten_and_unflatten(input_tree) -> Tuple[Callable, Callable]:
    """Make a pair of functions flatten(tree) -> x, unflatten(x) -> tree

    Intended usage:
    * Given a function f_structured(complicated_object), interface easily with
        optimizers and samplers that assume access to a differentiable function
        f_flat(x) where x is a flat Jax array

    See also:
    * HIPS autograd.misc.flatten
        https://github.com/HIPS/autograd/blob/c6f630a5ec18bd30f1485bc0dbbccb8664c77510/autograd/misc/flatten.py#L9-L15
        which has similar intended usage but a slightly different signature:
        flatten(value) -> flat_value, unflatten_fxn
    """

    tree_structure = tree_util.tree_structure(input_tree)
    leaf_shapes = [leaf.shape for leaf in tree_util.tree_leaves(input_tree)]

    def flatten(tree):
        leaves = tree_util.tree_leaves(tree)
        flattened_leaves = [leaf.flatten() for leaf in leaves]
        x = np.hstack(flattened_leaves)
        assert len(x.shape) == 1
        return x

    def unflatten(x):
        leaves = []
        i = 0
        for shape in leaf_shapes:
            num_elements = int(onp.prod(shape))
            leaves.append((x[i:i + num_elements].reshape(shape)))
            i += num_elements
        tree = tree_util.tree_unflatten(tree_structure, leaves)
        return tree

    return flatten, unflatten
