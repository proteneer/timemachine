import tensorflow as tf

class ConservativeForce():

    def params(self):
        raise NotImplementedError("Abstract base class")

    def energy(self, conf):
        raise NotImplementedError("Abstract base class")

    def gradients(self, conf):
        energy = self.energy(conf)
        return tf.gradients(energy, conf)[0]

    def hessians(self, conf):
        energy = self.energy(conf)
        return tf.hessians(energy, conf)[0]

    def mixed_partials(self, conf):
        # the order here matters since we're computing gradients and not
        # jacobians, otherwise they get reduced.
        mps = []
        energy = self.energy(conf)
        for p in self.params():
            mps.extend(tf.gradients(tf.gradients(energy, p), conf))
        return mps

# Energy functions that implement conservative forces.
class HarmonicBondForce(ConservativeForce):

    def __init__(self, precision=tf.float64):
        self.kb = tf.get_variable("kb", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(10.0))
        self.r0 = tf.get_variable("r0", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(1.2))
        self.b0 = tf.get_variable("b0", shape=tuple(), dtype=precision, initializer=tf.constant_initializer(0.0))

    def params(self):
        return [self.kb, self.r0, self.b0]

    def energy(self, conf):
        dx = tf.norm(conf[0]-conf[1])
        return self.kb*tf.pow(dx - self.r0 + self.b0, 2.0) 

if __name__ == "__main__":

    hb = HarmonicBondForce()
    conf = tf.convert_to_tensor([[1.0, 0.5, -0.5], [0.2, 0.1, -0.3]], dtype=tf.float64)

    e = hb.energy(conf)
    g = hb.gradients(conf)
    h = hb.hessians(conf)
    mp = hb.mixed_partials(conf)

    sess = tf.Session()
    sess.run(tf.initializers.global_variables())

    ee, gg, hh, mm = sess.run([e, g, h, mp])

    print(ee, gg.shape, hh.shape, mm)
