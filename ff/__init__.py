from ff.handlers import nonbonded, bonded

class Forcefield():
    """
    Utility class for wrapping around a list of ff_handlers
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
                assert self.hb_handle is None
                self.hb_handle = handle
            if isinstance(handle, bonded.HarmonicAngleHandler):
                assert self.ha_handle is None
                self.ha_handle = handle
            if isinstance(handle, bonded.ProperTorsionHandler):
                assert self.pt_handle is None
                self.pt_handle = handle
            if isinstance(handle, bonded.ImproperTorsionHandler):
                assert self.it_handle is None
                self.it_handle = handle
            if isinstance(handle, nonbonded.LennardJonesHandler):
                assert self.lj_handle is None
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

    def get_ordered_params(self):
        """
        Returns
        -------
        list of np.ndarray
            Return a flat, pre-determined ordering of the parameters
        """
        return [x.params for x in self.get_ordered_handles()]

    def get_ordered_handles(self):
        """
        Returns
        -------
        list of np.ndarray
            Return a flat, pre-determined ordering of the handlers
        """
        return [
            self.hb_handle,
            self.ha_handle,
            self.pt_handle,
            self.it_handle,
            self.q_handle,
            self.lj_handle
        ]