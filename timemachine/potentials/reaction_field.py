"""
References
----------
[Kubincová, Riniker, Hünenberger, 2020] Reaction-field electrostatics in molecular dynamics simulations:
    development of a conservative scheme compatible with an atomic cutoff
    https://pubs.rsc.org/en/content/articlelanding/2020/CP/D0CP03835K

Implementations:
* As OpenMM custom forces in Reeds
    https://github.com/rinikerlab/reeds/blob/2584f7d049f622df6a3acd1cc4216b14e404f01e/reeds/openmm/reeds_openmm.py#L229-L390
* As a proposed potential in Jax MD
    https://github.com/jax-md/jax-md/pull/199/files#diff-a0a92fb308e4969b7d9019f447d3a3871db8dd09e07db12170605ade70e77312R565-R585
"""

from jax import numpy as jnp

from timemachine.constants import ONE_4PI_EPS0


class RFConstants:
    def __init__(self, eps_rf=78.5, cutoff=1.2, mrf=4, nrf=6):
        """TODO: document"""
        self.eps_rf = eps_rf
        self.cutoff = cutoff
        self.mrf = mrf
        self.nrf = nrf

        self.krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff ** 3)
        self.arfm = (3 * cutoff ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * ((2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf))
        self.arfn = (3 * cutoff ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * ((2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf))
        self.crf = (
            ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + self.arfm * cutoff ** mrf + self.arfn * cutoff ** nrf
        )


def compute_rf_interaction_prefactors(q_j, r_ij, constants: RFConstants):
    return (
        ONE_4PI_EPS0
        * q_j
        * jnp.array(
            [
                1 / r_ij,
                constants.krf * r_ij ** 2,
                constants.arfm * r_ij ** 4,
                constants.arfn * r_ij ** 6,
                -constants.crf * jnp.ones_like(r_ij),
            ]
        )
    )  # TODO: optionally reduce over j


def rf_interaction(q_i, q_j, r_ij, constants: RFConstants):
    prefactors = compute_rf_interaction_prefactors(q_j, r_ij, constants)
    return jnp.sum(q_i * prefactors)
