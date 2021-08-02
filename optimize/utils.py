from jax import tree_util, numpy as np
import numpy as onp
from typing import Tuple, Callable


def get_shape(x):
    """safe to call on floats and arrays
    get_shape(1) -> None, get_shape(ones(2)) -> (2,)"""

    try:
        return x.shape
    except AttributeError:
        try:
            _ = float(x)
            return None
        except:
            raise ValueError("x doesn't appear to be an array or a float")


def reshape(x, shape):
    """assume x is scalar if shape is None"""
    if shape is None:
        return float(x)
    else:
        return x.reshape(shape)


def num_elements(shape):
    """floats (indicated by None) have 1 element"""
    if shape is None:
        return 1
    else:
        # note: length-zero arrays are fine in this branch: onp.prod(()) == 1.0
        return int(onp.prod(shape))


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

    leaf_shapes = [get_shape(leaf) for leaf in tree_util.tree_leaves(input_tree)]

    def flatten(tree):
        leaves = tree_util.tree_leaves(tree)
        flattened_leaves = [reshape(leaf, num_elements(get_shape(leaf))) for leaf in leaves]
        x = np.hstack(flattened_leaves)
        assert len(x.shape) == 1
        return x

    def unflatten(x):
        leaves = []
        i = 0
        for shape in leaf_shapes:
            leaves.append(reshape(x[i:i + num_elements(shape)], shape))
            i += num_elements
        tree = tree_util.tree_unflatten(tree_structure, leaves)
        return tree

    return flatten, unflatten
