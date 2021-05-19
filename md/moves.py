from md.states import CoordsVelBox
from typing import List, Tuple
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


class CompoundMove(MonteCarloMove):
    def __init__(self, moves: List[MonteCarloMove]):
        """Apply each of a list of moves in sequence"""
        self.moves = moves

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        for individual_move in self.moves:
            x = individual_move.move(x)
        return x

    @property
    def n_accepted_by_move(self):
        return np.array([m.n_accepted for m in self.moves])

    @property
    def n_proposed_by_move(self):
        return np.array([m.n_proposed for m in self.moves])

    @property
    def n_accepted(self):
        return np.sum(self.n_accepted_by_move)

    @property
    def n_proposed(self):
        return np.sum(self.n_proposed_by_move)
