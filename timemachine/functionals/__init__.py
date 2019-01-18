import tensorflow as tf

class Energy():

    def get_params(self):
        return self.params

    def total_params(self):
        """
        Returns total number of parameters across all dimensions.
        """
        tot = 0
        if isinstance(self.params, tf.Tensor):
            return self.params.get_shape().num_elements()

        for p in self.params:
            tot += p.get_shape().num_elements()
        return tot
