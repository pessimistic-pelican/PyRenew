
from __future__ import annotations

import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from pyrenew.metaclass import RandomVariable


class GammaObservation(RandomVariable):
    """Gamma observation"""

    def __init__(
        self,
        name: str,
        shape_rv: RandomVariable,
        eps: float = 1e-10,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        name
            Name for the numpyro variable.
        shape_rv
            Random variable from which to sample the positive shape parameter
            (sometimes called α or k) of the gamma distribution.
        eps
            Small value to add to the predicted mean to prevent numerical
            instability. Defaults to 1e-10.

        Returns
        -------
        None
        """

        GammaObservation.validate(shape_rv)

        self.name = name
        self.shape_rv = shape_rv
        self.eps = eps

    @staticmethod
    def validate(shape_rv: RandomVariable) -> None:
        """
        Check that the shape_rv is actually a RandomVariable

        Returns
        -------
        None
        """
        assert isinstance(shape_rv, RandomVariable)
        return None

    def sample(
        self,
        mu: ArrayLike,
        obs: ArrayLike | None = None,
        **kwargs,
    ) -> ArrayLike:
        """
        Sample from the gamma distribution

        Parameters
        ----------
        mu
            Mean parameter of the gamma distribution.
        obs
            Observed data, by default None.
        **kwargs
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        ArrayLike
        """
        shape = self.shape_rv.sample()

        # numpyro uses shape (concentration) and rate parameterisation
        gamma_sample = numpyro.sample(
            name=self.name,
            fn=dist.Gamma(
                concentration=shape,
                rate=shape / (mu + self.eps),
            ),
            obs=obs,
        )

        return gamma_sample