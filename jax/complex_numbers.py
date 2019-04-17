from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import numpy as onp

import tensorflow as tf

zs = 0.5j * np.arange(5) + np.arange(5)

print("input", zs)


def fn(z):
    return np.cos(np.linalg.norm(z*2))

grad = jax.jacfwd(fn)
print("jax", fn(zs), grad(zs))

def tf_fn(z):
    return tf.cos(tf.norm(z*2))

tf_zs = tf.convert_to_tensor(0.5j * onp.arange(5) + onp.arange(5))
tf_res = tf_fn(tf_zs)

sess = tf.Session()


grad_ys = tf.ones_like(tf_res)
grad_op = tf.gradients(tf_res, tf_zs, grad_ys=grad_ys)
print("tf", sess.run([tf_res, grad_op, grad_ys]))