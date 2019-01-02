import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian


class ConservativeForce():
    """
    A conservative force has an integral equal to a scalar-valued energy function.
    """

    def get_params(self):
        return self.params

    def total_params(self):
        """
        Returns total number of parameters across all dimensions.
        """
        tot = 0
        for p in self.params:
            tot += p.get_shape().num_elements()
        return tot

    def energy(self, conf):
        """
        Computes a scalar energy given a geometry.
        """
        raise NotImplementedError("Abstract base class")

    def gradients(self, conf):
        energy = self.energy(conf)
        gradients = tf.gradients(energy, conf)[0]
        return gradients

    def hessians(self, conf):
        energy = self.energy(conf)
        return tf.hessians(energy, conf)[0]

    def mixed_partials(self, conf):
        """
        Returns list of tensors of mixed partial derivatives evaluated at the input geometry.

        Parameters
        ----------
        conf: tf.placeholder
            An N x 3 configuration placeholder

        Returns
        -------
        tf.Tensor of size len(self.params)
            Returns an unflattened list of mixed partial derivatives [(p_shape), N, 3]
            matching each parameter in get_params()

        """
        # (ytz): Note for implementation purposes, the order of differentiation
        # actually matters. The jacobian system in tensorflow expects a fixed size
        # tensor for the outputs, while permitting a variable list of tensors for 
        # inputs. This means that we should naturally use the coordinate derivatives
        # as they all have a fixed N x 3 structure, where as the input parameters
        # can take on a variadic list of tensors of varying sizes.

        # optimized version to speed things up a little bit.
        grads = self.gradients(conf)

        # taken from tf src gradients_impl.py _IndexedSlicesToTensor 
        if isinstance(grads, tf.IndexedSlices):
            grads = tf.unsorted_segment_sum(
                grads.values,
                grads.indices,
                grads.dense_shape[0])

        reverse_shaped = jacobian(grads, self.params, use_pfor=False) 

        if isinstance(reverse_shaped, tf.Tensor):
            # shove to a list
            reverse_shaped = [reverse_shaped]

        properly_shaped = []
        for p in reverse_shaped:
            if len(p.get_shape()) == 2:
                properly_shaped.append(p) # already ready to go
            elif len(p.get_shape()) == 3:
                properly_shaped.append(tf.transpose(p, perm=(2,0,1)))
            elif len(p.get_shape()) == 4:
                # properly_shaped.append(tf.reshape(fixed, [-1, fixed.shape[2], fixed.shape[3]]))
                properly_shaped.append(tf.transpose(p, perm=(2,3,0,1)))
            else:
                # should be easy to support, just add perm=(2,3,...,0,1)
                raise NotImplementedError("Shapes > 4 not supported")
        return properly_shaped
