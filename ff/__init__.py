from ff.handlers import nonbonded, bonded

class Forcefield():
    """
    A forcefield supports generation of parameters of a standard
    molecular system.
    """


    def __init__(self, ff_handlers):
        self.hb_handle = None
        self.ha_handle = None
        self.pt_handle = None
        self.it_handle = None
        self.lj_handle = None
        self.q_handle = None
        for handle in ff_handlers:
            if isinstance(handle, bonded.HarmonicBondHandler):
                self.hb_handle = handle
            if isinstance(handle, bonded.HarmonicAngleHandler):
                self.ha_handle = handle
            if isinstance(handle, bonded.ProperTorsionHandler):
                self.pt_handle = handle
            if isinstance(handle, bonded.ImproperTorsionHandler):
                self.it_handle = handle
            if isinstance(handle, nonbonded.LennardJonesHandler):
                self.lj_handle = handle
            if isinstance(handle, nonbonded.AM1CCCHandler):
                assert self.q_handle is None
                self.q_handle = handle
            if isinstance(handle, nonbonded.AM1BCCHandler):
                assert self.q_handle is None
                self.q_handle = handle
            if isinstance(handle, nonbonded.SimpleChargeHandler):
                assert self.q_handle is None
                self.q_handle = handle

    # def serialize(self):


    # def deserialize(self):