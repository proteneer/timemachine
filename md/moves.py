from md.states import CoordsVelBox
from typing import Tuple
import numpy as np


class MonteCarloMove:
    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        """ return proposed state and log acceptance probability """
        raise NotImplementedError

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        proposal, log_acceptance_probability = self.propose(x)
        self.n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            return proposal
        else:
            return x

    @property
    def acceptance_fraction(self):
        if self.n_proposed > 0:
            return self.n_accepted / self.n_proposed
        else:
            return 0.0
