# numpydoc ignore=GL08

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import numpyro
from jax.typing import ArrayLike

import pyrenew.arrayutils as au
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable


class SeverePrimarySample(NamedTuple):
    """
    A container to hold the output of
    [`SeverePrimary.sample`][].

    Attributes
    ----------
    alpha_severe : ArrayLike, optional
        The infection-to-severe rate. Defaults to None.
    alpha_primary : ArrayLike, optional
        The infection-to-primary care case rate. Defaults to None.
    latent_severe : ArrayLike or None
        The computed number of severe admissions. Defaults to None.
    latent_primary : ArrayLike or None
        The computed number of primary care cases. Defaults to None.
    multiplier : ArrayLike or None
        The day of the week effect multiplier. Defaults to None. It
        should match the number of timepoints in the latent hospital
        admissions.
    """

    alpha_severe: ArrayLike | None = None
    alpha_primary: ArrayLike | None = None
    latent_severe: ArrayLike | None = None
    latent_primary: ArrayLike | None = None
    multiplier: ArrayLike | None = None

    def __repr__(self):
        return f"SeverePrimarySample(alpha_severe={self.alpha_severe},alpha_primary={self.alpha_primary})"


class SeverePrimary(RandomVariable):
    r"""
    Latent Severe + Primary (Necessary to share a single symptomatic random variable)

    Implements a renewal process for the expected number of severe cases and primary care cases.

    Notes
    -----
    The following text was directly extracted from the wastewater model
    documentation (`link <https://github.com/CDCgov/ww-inference-model/blob/main/model_definition.md#hospital-admissions-component>`_).

    Following other semi-mechanistic renewal frameworks, we model the *expected*
    hospital admissions per capita $H(t)$ as a convolution of the
    *expected* latent incident infections per capita $I(t)$, and a
    discrete infection to hospitalization distribution $d(\tau)$, scaled
    by the probability of being hospitalized $p_\mathrm{hosp}(t)$.

    To account for day-of-week effects in hospital reporting, we use an
    estimated *day of the week effect* $\omega(t)$. If $t$ and $t'$
    are the same day of the week, $\omega(t) = \omega(t')$. The seven
    values that $\omega(t)$ takes on are constrained to have mean 1.

    ```math
    H(t) = \omega(t) p_\mathrm{hosp}(t) \sum_{\tau = 0}^{T_d} d(\tau) I(t-\tau)
    ```

    Where $T_d$ is the maximum delay from infection to hospitalization
    that we consider.
    """

    def __init__(
        self,
        infection_to_severe_interval_rv: RandomVariable,
        infection_to_primary_interval_rv: RandomVariable,
        infection_severe_ratio_rv: RandomVariable,
        infection_primary_ratio_rv: RandomVariable,
        day_of_week_effect_rv: RandomVariable | None = None,
        symptomatic_ratio_rv: RandomVariable | None = None,
        obs_data_first_day_of_the_week: int = 0,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        infection_to_severe_interval_rv
            pmf for reporting (informing) severe admissions (see
            pyrenew.observations.Deterministic).
                infection_to_severe_interval_rv
            pmf for reporting (informing) primary care cases
        infection_severe_ratio_rv
            Infection to severe rate random variable.
        infection_primary_ratio_rv
            Infection to primary care rate random variable.
        day_of_week_effect_rv
            Day of the week effect. Should return a ArrayLike with 7
            values. Defaults to a deterministic variable with
            jax.numpy.ones(7) (no effect).
        symptomatic_ratio_rv
            Random variable for the symptomatic
            probability. Defaults to 1 (full reporting).
        obs_data_first_day_of_the_week
            The day of the week that the first day of the observation data
            corresponds to. Valid values are 0-6, where 0 is Monday and 6 is
            Sunday. Defaults to 0.

        Returns
        -------
        None
        """

        if day_of_week_effect_rv is None:
            day_of_week_effect_rv = DeterministicVariable(
                name="weekday_effect", value=jnp.ones(7)
            )

        SeverePrimary.validate(
            infection_to_severe_interval_rv,
            infection_to_primary_interval_rv,
            infection_severe_ratio_rv,
            infection_primary_ratio_rv,
            day_of_week_effect_rv,
            symptomatic_ratio_rv,
            obs_data_first_day_of_the_week,
        )

        self.infection_to_severe_interval_rv = infection_to_severe_interval_rv
        self.infection_to_primary_interval_rv = infection_to_primary_interval_rv
        self.infection_severe_ratio_rv = infection_severe_ratio_rv
        self.infection_primary_ratio_rv = infection_primary_ratio_rv
        self.day_of_week_effect_rv = day_of_week_effect_rv
        self.symptomatic_ratio_rv = symptomatic_ratio_rv
        self.obs_data_first_day_of_the_week = obs_data_first_day_of_the_week

    @staticmethod
    def validate(
        infection_to_severe_interval_rv: Any,
        infection_to_primary_interval_rv: Any,
        infection_severe_ratio_rv: Any,
        infection_primary_ratio_rv: Any,
        day_of_week_effect_rv: Any,
        symptomatic_ratio_rv: Any,
        obs_data_first_day_of_the_week: Any,
    ) -> None:
        """
        Validates that the IHR, weekday effects, probability of being
        reported hospitalized distributions, and infection to
        hospital admissions reporting delay pmf are RandomVariable types

        Parameters
        ----------
        infection_to_severe_interval_rv
            Possibly incorrect input for the infection to severe
            interval distribution.
        infection_to_primary_interval_rv
            Possibly incorrect input for the infection to primary care
            interval distribution.
        infection_severe_ratio_rv
            Possibly incorrect input for infection to severe rate distribution.
        infection_primary_ratio_rv
            Possibly incorrect input for infection to severeprimary rate distribution.
        day_of_week_effect_rv
            Possibly incorrect input for day of the week effect.
        symptomatic_ratio_rv
            Possibly incorrect input for distribution or fixed value for the
            symptomatic probability
        obs_data_first_day_of_the_week
            Possibly incorrect input for the day of the week that the first day
            of the observation data corresponds to. Valid values are 0-6, where
            0 is Monday and 6 is Sunday.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If any of the random variables are not of the correct type, or if
            the day of the week is not within the valid range.
        """
        assert isinstance(infection_to_severe_interval_rv, RandomVariable)
        assert isinstance(infection_to_primary_interval_rv, RandomVariable)
        assert isinstance(infection_severe_ratio_rv, RandomVariable)
        assert isinstance(infection_primary_ratio_rv, RandomVariable)
        assert isinstance(day_of_week_effect_rv, RandomVariable)
        assert isinstance(symptomatic_ratio_rv, RandomVariable)
        assert isinstance(obs_data_first_day_of_the_week, int)
        assert 0 <= obs_data_first_day_of_the_week <= 6

        return None

    def sample(
        self,
        latent_infections: ArrayLike,
        n: int,
        weekly_obs: bool = None,
        **kwargs,
    ) -> SeverePrimarySample:
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
        SeverePrimarySample
        """

        p_sym = self.symptomatic_ratio_rv(**kwargs)

        infection_severe_rate = self.infection_severe_ratio_rv(p_sym = p_sym, n = n,**kwargs)
        infection_primary_rate = self.infection_primary_ratio_rv(p_sym = p_sym, n = n,**kwargs)

        infection_to_severe_interval = self.infection_to_severe_interval_rv(**kwargs)
        infection_to_primary_interval = self.infection_to_primary_interval_rv(**kwargs)

        latent_severe_admissions, _ = compute_delay_ascertained_incidence( ## takes size n + n_initialisation and turn it into size n if inf_to_adm interval is size n_initialisation
            latent_infections,
            infection_to_severe_interval,
            infection_severe_rate,
        )

        latent_primary_care, _ = compute_delay_ascertained_incidence( ## takes size n + n_initialisation and turn it into size n if inf_to_adm interval is size n_initialisation
            latent_infections,
            infection_to_primary_interval,
            infection_primary_rate,
        )
        # Applying the day of the week effect. For this we need to:
        # 1. Get the day of the week effect
        # 2. Identify the offset of the latent_infections
        # 3. Apply the day of the week effect to the
        # latent_hospital_admissions
        dow_effect_sampled = self.day_of_week_effect_rv(**kwargs)

        if dow_effect_sampled.size != 7:
            raise ValueError(
                "Day of the week effect should have 7 values. "
                f"Got {dow_effect_sampled.size} instead."
            )

        inf_offset = self.obs_data_first_day_of_the_week % 7

        # Replicating the day of the week effect to match the number of
        # timepoints
        dow_effect = au.tile_until_n(
            data=dow_effect_sampled,
            n_timepoints=latent_primary_care.size, ###TBC
            offset=inf_offset,
        )

        # latent_hospital_admissions = latent_hospital_admissions * dow_effect


        if weekly_obs is True:
            # Aggregate daily latent hospital admissions into weekly buckets.
            # n is the number of (daily) timepoints. Expect n to be a multiple of 7
            # when weekly_obs is requested. Compute number of weeks and reshape
            # into (n_weeks, 7) then sum across the week axis.
            n_weeks = n // 7
            # reshape to (n_weeks, 7) and sum across days to get weekly totals
            latent_severe_admissions = latent_severe_admissions[-n:].reshape((n_weeks, 7)).sum(axis=1)
            latent_primary_care = latent_primary_care[-n:].reshape((n_weeks, 7)).sum(axis=1)


        numpyro.deterministic("latent_severe_admissions", latent_severe_admissions)
        numpyro.deterministic("latent_primary_care", latent_primary_care)
        numpyro.deterministic("infection_severe_rate", infection_severe_rate)
        numpyro.deterministic("infection_primary_rate", infection_primary_rate)
        
        return SeverePrimarySample(
            alpha_severe=infection_severe_rate,
            alpha_primary=infection_primary_rate,
            latent_severe=latent_severe_admissions,
            latent_primary=latent_primary_care,
            multiplier=dow_effect,
        )
