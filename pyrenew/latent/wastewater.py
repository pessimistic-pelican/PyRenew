# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

import pyrenew.arrayutils as au
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class WastewaterSample(NamedTuple):
    """
    A container to hold the output of
    [`Wastewater.sample`][].

    Attributes
    ----------
    alpha : ArrayLike, optional
        The time-varying average genome copies shed per person. Defaults to None.
    latent_genome_copies 
    The computed genome copies. Defaults to None.
    """

    alpha: ArrayLike | None = None
    latent_genome_copies: ArrayLike | None = None

    def __repr__(self):
        return f"WastewaterSample(alpha={self.alpha}, latent_genome_copies={self.latent_genome_copies})"


class Wastewater(RandomVariable):
    r"""
    Latent genome copies

    Implements a renewal process for the expected number of genome copies in the four reclaimation plants.

    Notes
    -----
    Following other semi-mechanistic renewal frameworks, we model the *expected*
    genome copies per capita $H(t)$ as a convolution of the
    *expected* latent incident infections per capita $I(t)$, and a
    discrete infection to shedding distribution $d(\tau)$, scaled
    by a time-varying average genome copies shed per person.
    """

    def __init__(
        self,
        infection_to_shedding_rv: RandomVariable,
        alpha_rv: RandomVariable
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        infection_to_shedding_rv
            pmf for infection to shedding rv
        alpha_rv
            total average genome copies shed per person
        
        Returns
        -------
        None
        """

        Wastewater.validate(
            infection_to_shedding_rv,
            alpha_rv,
        )
        
        self.infection_to_shedding_rv = infection_to_shedding_rv
        self.alpha_rv = alpha_rv
        
    @staticmethod
    def validate(
        infection_to_shedding_rv: Any,
        alpha_rv: Any,

    ) -> None:
        """
        Validates that the wastewater variables as RandomVariable

        Parameters
        ----------
        infection_to_shedding_rv
            Possibly incorrect input for the infection to shedding
            interval distribution.
        alpha_rv
            Possibly incorrect input for alpha.
        
        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If any of the random variables are not of the correct type, or if
            the day of the week is not within the valid range.
        """
        assert isinstance(infection_to_shedding_rv, RandomVariable)
        assert isinstance(alpha_rv, RandomVariable)
        
        return None

    def sample(
        self,
        latent_infections: ArrayLike,
        n: int,
        **kwargs,
    ) -> WastewaterSample:
        """
        Samples from the observation process

        Parameters
        ----------
        latent_infections
            Latent infections.
        **kwargs
            Additional keyword arguments passed through to
            internal `sample()` calls,
            should there be any.

        Returns
        -------
        WastewaterSample
        """

        alpha = self.alpha_rv(n=n,**kwargs)

        infection_to_shedding_interval = self.infection_to_shedding_rv(
            **kwargs
        )

        latent_genome_copies, _ = compute_delay_ascertained_incidence(
            latent_infections,
            infection_to_shedding_interval,
            p_observed_given_incident = alpha
        )

        numpyro.deterministic("latent_genome_copies", latent_genome_copies)
        numpyro.deterministic("alpha", alpha)

        return WastewaterSample(
            alpha=alpha,
            latent_genome_copies=latent_genome_copies,
        )
