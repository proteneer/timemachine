import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian


class ConservativeForce():
    """
    A conservative force has an integral equal to a scalar-valued energy function.
    """

    def get_params(self):
        return self.params

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
        energy = self.energy(conf)
        params = self.get_params()
        dEdp = tf.stack(tf.gradients(energy, params))
        return jacobian(dEdp, conf, use_pfor=False) # use_pfor doesn't work with SparseTensors

        # (ytz) TODO: saving an old copy for pedagogy
        # mps = []
        # energy = self.energy(conf)
        # for p in self.get_params():
        #     mps.extend(tf.gradients(tf.gradients(energy, p), conf))
        # return mps
