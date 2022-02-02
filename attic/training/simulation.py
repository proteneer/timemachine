class Simulation:
    def __init__(self, x, v, box, potentials, integrator):
        self.x = x
        self.v = v
        self.box = box
        self.potentials = potentials
        self.integrator = integrator
