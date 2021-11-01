from md.states import CoordsVelBox
from typing import List, Tuple
import numpy as np


class MonteCarloMove:
    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        """return proposed state and log acceptance probability"""
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


from scipy.special import logsumexp


class MultipleTryMetropolis:
    def __init__(self, K):
        self.K = K
        self.n_accepted = 0
        self.n_proposed = 0

    def batch_propose(self, x: CoordsVelBox):
        raise NotImplementedError()

    def batch_log_prob(self, proposals):
        raise NotImplementedError()

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        self.n_proposed += 1
        # assume proposals themselves have water and ligand aligned and combined
        numerator_proposals = self.batch_propose(x)
        numerator_log_probs = self.batch_log_prob(numerator_proposals)
        assert self.K == len(numerator_proposals)
        assert self.K == len(numerator_log_probs)
        normalized_numerator_log_probs = np.exp(numerator_log_probs - logsumexp(numerator_log_probs))

        new_sample = numerator_proposals[np.random.choice(np.arange(self.K), p=normalized_numerator_log_probs)]

        denominator_proposals = self.batch_propose(new_sample)
        denominator_proposals.append(x)

        denominator_log_probs = self.batch_log_prob(denominator_proposals)

        log_ratio = logsumexp(numerator_log_probs) - logsumexp(denominator_log_probs)

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_ratio)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            return new_sample
        else:
            return x
