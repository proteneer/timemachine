# contains approaches to make MCMC moves "gradual"

# classic references:
# * [Neal, 1996] Sampling from multimodal distributions using tempered transitions
#       https://link.springer.com/article/10.1007/bf00143556
# * [Nilmeier, Crooks, Minh, Chodera, 2011] Nonequilibrium candidate Monte Carlo
#       is an efficient tool for equilibrium simulation
#       https://www.pnas.org/doi/full/10.1073/pnas.1106094108


from typing import Sequence

import numpy as np
from scipy.stats import norm


# sometimes, we might have a cheaper way to compute logpdf(x) - logpdf(y)
# (that doesn't involve calling logpdf twice)
class Target:
    def unnormalized_logpdf(self, x):
        raise NotImplementedError

    def logpdf_difference(self, x, y):
        """logpdf(x) - logpdf(y)"""
        return self.unnormalized_logpdf(x) - self.unnormalized_logpdf(y)


class ConditionalDistribution:
    def sample(self, x):
        """sample y ~ p(. | x)"""
        raise NotImplementedError

    def logpdf(self, x, y):
        """evaluate log p(y | x)"""
        raise NotImplementedError


class GaussianRandomWalkProposal(ConditionalDistribution):
    def __init__(self, sig=0.01, seed=None):
        self.sig = sig
        if seed is None:
            seed = np.random.randint(1000000)
        self.rng = np.random.default_rng(seed)

    def sample(self, x):
        return self.rng.normal(loc=x, scale=self.sig, size=x.shape)

    def logpdf(self, x, y):
        return np.sum(norm.logpdf(y, loc=x, scale=self.sig))


class PathConditionalDistribution:
    def __init__(self, proposals: Sequence[ConditionalDistribution]):
        """Defines a conditional distribution on (x_1, x_2, ..., x_T) given x_0
        using a sequence of T conditional distributions p_t(x_{t+1}| x_t)"""
        self.proposals = proposals

    def sample(self, x):
        path = [x]
        for proposal in self.proposals:
            path.append(proposal.sample(path[-1]))
        return path

    def logpdf(self, path):
        """Conditional pdf(x) = prod_t p_t(x_{t+1} | x_t)"""
        assert len(path) == len(self.proposals) + 1
        log_increments = []
        for t in range(len(self.proposals)):
            log_increments.append(self.proposals[t].logpdf(path[t], path[t + 1]))
        return np.sum(log_increments)


class PathwiseMH:
    def __init__(self, path_proposal: PathConditionalDistribution, target: Target):
        """EXPERIMENTAL

        Special case of NCMC, without any perturbation steps.

        CAVEATS:
        * not well-tested yet!
        * acceptance rate of this variant can approach 0 in limit of long paths(!)

        Motivations:
        * Possible simplicity -- may be able to avoid defining a sequence of intermediate distributions
        * Possible efficiency -- may be able to get away with fewer energy/gradient evaluations,
            if using trajectories that are much too short for a typical NCMC implementation
        """
        self.path_proposal = path_proposal
        self.rev_path_proposal = PathConditionalDistribution(self.path_proposal.proposals[::-1])
        self.target = target

    def move(self, x):
        path = self.path_proposal.sample(x)
        y = path[-1]

        favorability = self.target.logpdf_difference(y, x)
        rev_path = path[::-1]
        irreversibility = self.path_proposal.logpdf(path) - self.rev_path_proposal.logpdf(rev_path)

        log_accept_prob = np.minimum(0.0, favorability - irreversibility)
        accept_prob = np.exp(log_accept_prob)
        accepted = np.random.rand() < accept_prob

        # log stuff
        aux = dict(
            proposal_path=path,
            favorability=favorability,
            irreversibility=irreversibility,
            log_accept_prob=log_accept_prob,
            accepted=accepted,
        )

        if accepted:
            y_next = y
        else:
            y_next = x
        return y_next, aux


class ReversibleNCMCMove:
    def __init__(self, propagators, logpdf_difference_fxns):
        """Common special case of NCMC / tempered transitions / ...,
        where we have defined a sequence of intermediate distributions
        that start and end at the target distribution,
        and we apply MCMC transitions that target each of these distributions in turn.

        This implementation ASSUMES WITHOUT CHECKING that:
        * there exists a sequence of distributions p[0], p[1], ..., p[T]
            such that p[0] = p[T] = p_target
        * each propagators[i] is a reversible MCMC move targeting p[i]
        * each logpdf_difference_fxns[i](x) implements log_p[i](x) - log_p[i-1](x)

        """
        assert len(propagators) == len(logpdf_difference_fxns)
        self.propagators = propagators
        self.logpdf_difference_fxns = logpdf_difference_fxns

    def move(self, x):
        proposal_traj = [x]  # TODO: maybe avoid unnecessary memory cost?
        incremental_logpdf_diffs = []
        for (diff, prop) in zip(self.logpdf_difference_fxns, self.propagators):
            incremental_logpdf_diffs.append(diff(proposal_traj[-1]))
            proposal_traj.append(prop.move(proposal_traj[-1]))
        log_accept_prob = np.minimum(0.0, np.sum(incremental_logpdf_diffs))
        accept_prob = np.exp(log_accept_prob)
        accepted = np.random.rand() < accept_prob

        # log stuff
        aux = dict(
            proposal_path=proposal_traj,
            incremental_logpdf_diffs=incremental_logpdf_diffs,
            log_accept_prob=log_accept_prob,
            accepted=accepted,
        )

        if accepted:
            return proposal_traj[-1], aux
        else:
            return x, aux
