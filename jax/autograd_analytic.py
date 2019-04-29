import numpy as onp
import autograd as ag
import autograd.numpy as anp
import tensorflow as tf

rs = anp.array([2.0, 3.0])
qs = anp.array([-0.5, 0.3])
k = 0.5



def ag_fn(r):

    r_i = r[0]
    r_j = r[1]

    q_i = qs[0]
    q_j = qs[1]

    Sk = q_i*anp.exp(1j*anp.dot(k, r_i)) + q_j*anp.exp(1j*anp.dot(k, r_j))

    return anp.power(anp.abs(Sk), 2)


ag_hess = ag.hessian(ag_fn)

print("ag val:", ag_fn(rs))
print("ag hess:", ag_hess(rs))

# def tf_fn(x):
#     real = tf.cos(x+2)
#     imag = tf.sin(x-1)
#     return tf.abs(tf.complex(real, imag))

# # tf_inp = tf.convert_to_tensor(inp)
# tf_inp = tf.placeholder(shape=tuple(), dtype=onp.float64)

# out_op = tf_fn(tf_inp)

# tf_grad = tf.gradients(out_op, tf_inp)[0]
# tf_hess = tf.hessians(out_op, tf_inp)[0]

# sess = tf.Session()
# delta = 1e-7

# _, d0, tf_ad = sess.run([out_op, tf_grad, tf_hess], feed_dict={tf_inp: inp})
# _, d1, _ = sess.run([out_op, tf_grad, tf_hess], feed_dict={tf_inp: inp+delta})

# print("numerical derivative:", (d1-d0)/delta)
# print("tf_autodiff derivative:", tf_ad)