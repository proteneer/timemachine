# MC moves for buried water sampling

import numpy as np
from scipy.stats import norm

from timemachine.md.gradual import ConditionalDistribution


class WaterDisplacementProposal(ConditionalDistribution):
    def __init__(self, water_idxs, sig=1.0, seed=None):
        """Pick a random water, and rigidly translate it by disp ~ N(zeros(3), sig)"""
        for w in water_idxs:
            assert len(w) == 3, "expected each water to have 3 particles"
        self.water_idxs = water_idxs
        self.n_waters = len(self.water_idxs)
        self._water_sets = [set(w) for w in water_idxs]

        self.sig = sig
        if seed is None:
            seed = np.random.randint(1000000)
        self.rng = np.random.default_rng(seed)

    def sample(self, x):

        # pick which water to move
        i = self.rng.integers(0, self.n_waters)
        idxs = self.water_idxs[i]

        # sample a displacement vector
        disp = self.rng.normal(loc=0, scale=self.sig, size=3)

        y = np.array(x)
        y[idxs] += disp
        return y

    def logpdf(self, x, y, assume_valid=False):
        """Note: assumes that y hasn't been re-imaged to home-box when computing displacement y - x.

        If assume_valid==False, will verify (inefficiently!) that
            only a single water has been updated,
            and that the update was a rigid translation.
        """
        all_displacements = y - x
        move_mask = np.linalg.norm(all_displacements, axis=1) > 0
        _disp = y[move_mask] - x[move_mask]
        disp = np.mean(_disp, 0)

        if not assume_valid:
            # only a valid move if a single water was moved
            atoms_that_moved = set(np.nonzero(move_mask))
            if len(atoms_that_moved) > 0:
                valid = sum((atoms_that_moved.issubset(s)) for s in self._water_sets) <= 1
            else:
                valid = True

            # further, all moved water particles could have only been updated via single shared displacement vector
            valid = valid and np.std(_disp, 0) < 1e-5
            if not valid:
                return -np.inf

        discrete_choice_logprob = -np.log(self.n_waters)
        displacement_logprob = np.sum(norm.logpdf(disp, scale=self.sig))

        return discrete_choice_logprob + displacement_logprob
