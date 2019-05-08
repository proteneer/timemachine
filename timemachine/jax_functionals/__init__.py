import jax

class Energy():

    def __init__(self):
        self._gradient_fn = None

    def energy(self, conf, params, box=None):
        """
        All subclasses must implement a method that takes in
        a geometry and an opaque handle to a set of parameters
        """
        raise NotImplementedError()

    def gradient(self, conf, params, box=None, *args, **kwargs):
        if self._gradient_fn is None:
            # (ytz): jacrevs are more efficient than jacfwds
            # for first order derivatives
            self._gradient_fn = jax.jit(jax.jacrev(self.energy, 0))
            # self._gradient_fn = jax.jacrev(self.energy, 0)

        return self._gradient_fn(conf, params, box, *args, **kwargs)
