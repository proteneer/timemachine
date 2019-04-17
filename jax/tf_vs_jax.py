import numpy as onp
import autograd as ag
import autograd.numpy as anp
import numpy as onp
import tensorflow as tf

inp = anp.array(2.0)

print("input", inp)

def ag_fn(x):
    real = anp.cos(x+2)
    imag = anp.sin(x-1)
    return anp.abs(real+1j*imag)

ag_hess = ag.hessian(ag_fn)

print("ag val:", ag_fn(inp))
print("ag hess:", ag_hess(inp))

def tf_fn(x):
    real = tf.cos(x+2)
    imag = tf.sin(x-1)
    return tf.abs(tf.complex(real, imag))

# tf_inp = tf.convert_to_tensor(inp)
tf_inp = tf.placeholder(shape=tuple(), dtype=onp.float64)

out_op = tf_fn(tf_inp)

tf_grad = tf.gradients(out_op, tf_inp)[0]
tf_hess = tf.hessians(out_op, tf_inp)[0]

sess = tf.Session()
delta = 1e-7

_, d0, tf_ad = sess.run([out_op, tf_grad, tf_hess], feed_dict={tf_inp: inp})
_, d1, _ = sess.run([out_op, tf_grad, tf_hess], feed_dict={tf_inp: inp+delta})

print("numerical derivative:", (d1-d0)/delta)
print("tf_autodiff derivative:", tf_ad)