from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
import jax.numpy as jnp

from pyrenew.metaclass import RandomVariable


class LogNormalObservation(RandomVariable):
    """LogNormal observation"""

    def __init__(
        self,
        name: str,
        sigma_rv: RandomVariable,
        eps: float = 1e-10,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name : str
            Name for the numpyro variable.
        sigma_rv : RandomVariable
            Random variable from which to sample the positive scale parameter
            (standard deviation on the log scale) of the log-normal.
        eps : float, optional
            Small value to add to the predicted mean before taking log to
            prevent numerical instability. Defaults to 1e-10.

        Returns
        -------
        None
        """
        LogNormalObservation.validate(sigma_rv)

        self.name = name
        self.sigma_rv = sigma_rv
        self.eps = eps

    @staticmethod
    def validate(sigma_rv: RandomVariable) -> None:
        """
        Check that sigma_rv is a RandomVariable.

        Parameters
        ----------
        sigma_rv : RandomVariable
            Random variable from which to sample the positive log-scale
            standard deviation parameter of the log-normal distribution.

        Returns
        -------
        None
        """
        assert isinstance(sigma_rv, RandomVariable)
        return None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> ArrayLike:
        """
        Sample from the log-normal distribution.

        Parameters
        ----------
        mu : ArrayLike
            Mean parameter of the log-normal on the **natural scale**.
            Must be strictly positive.
        obs : ArrayLike, optional
            Observed data (positive values). Defaults to None.
        **kwargs
            Additional keyword arguments passed through to internal sample calls.

        Returns
        -------
        ArrayLike
            Sampled values from the log-normal.
        """
        sigma = self.sigma_rv.sample()

        lognormal_sample = numpyro.sample(
            name=self.name,
            fn=dist.LogNormal(
                loc=jnp.log(mu + self.eps),  # log-mean
                scale=sigma,                 # std dev on log scale
            ),
            obs=obs,
        )

        return lognormal_sample
