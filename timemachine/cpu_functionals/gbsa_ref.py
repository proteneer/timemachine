import tensorflow as tf


def compute_radii(xs):
    xi = tf.expand_dims(xs, 1)
    xj = tf.expand_dims(xs, 0)
    return tf.reduce_sum(tf.pow(xi - 0.5*xj, 2), axis=-1)


def outer_loop(xs):
    rs = compute_radii(xs)
    ri = tf.expand_dims(rs, axis=1)
    rj = tf.expand_dims(rs, axis=0)
    d2ij = tf.pow(ri - 1.5*rj, 2)
    nrg = tf.reduce_sum(d2ij)
    return nrg, tf.gradients(nrg, rs)


if __name__ == "__main__":
    sess = tf.Session()
    # xs = tf.convert_to_tensor([1.0, 2.3, 0.4, -0.3, 1.2])
    xs = tf.convert_to_tensor([1.0, 2.3, 0.4], dtype=tf.float64)
    # xs = tf.convert_to_tensor([1.0, 2.3], dtype=tf.float64)
    radii_op = compute_radii(xs)
    nrg_op, dE_drs = outer_loop(xs)

    grad_op = tf.gradients(nrg_op, xs)
    
    dRdxi = tf.gradients(radii_op[0], xs)
    dRdxj = tf.gradients(radii_op[1], xs)

    # print(sess.run([radii_op, dE_drs, dRdxi, dRdxj]))

    print(sess.run([grad_op]))
