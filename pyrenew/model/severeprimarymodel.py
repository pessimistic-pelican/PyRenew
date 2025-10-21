# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

from jax.typing import ArrayLike

from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel


class SeverePrimarySample(NamedTuple):
    """
    A container for holding the output from `model.SeverePrimarySample.sample()`.

    Attributes
    ----------
    Rt : ArrayLike | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated number of new infections over time. Defaults to None.
    infection_severe_rate : ArrayLike | None, optional
        The infected hospitalization rate. Defaults to None.
    infection_primary_rate : ArrayLike | None, optional
        The infected to primary care case rate. Defaults to None.
    latent_severe_primary : ArrayLike | None, optional
        The estimated latent severe cases and primary cases. Defaults to None.
    observed_severe_admissions : ArrayLike | None, optional
        The sampled or observed hospital admissions. Defaults to None.
    observed_primary_care_cases : ArrayLike | None, optional
        The sampled or observed primary care flu+ cases. Defaults to None.
    """

    Rt: ArrayLike | None = None
    latent_infections: ArrayLike | None = None
    infection_severe_rate: ArrayLike | None = None
    infection_primary_rate: ArrayLike | None = None
    latent_severe_admissions: ArrayLike | None = None
    latent_primary_care_cases: ArrayLike | None = None
    observed_severe_admissions: ArrayLike | None = None
    observed_primary_care_cases: ArrayLike | None = None

    def __repr__(self):
        return (
            f"HospModelSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"infection_severe_rate={self.infection_severe_rate}, "
            f"infection_primary_rate={self.infection_primary_rate}, "
            f"latent_severe_admissions={self.latent_severe_admissions}, "
            f"latent_primary_care_cases={self.latent_primary_care_cases}, "
            f"observed_primary_care_cases={self.observed_severe_admissions}, "
            f"observed_severe_admissions={self.observed_severe_admissions}"
        )


class SeverePrimaryModel(Model):
    """
    Severe Cases + Primary care cases Model (BasicRenewal + Severe cases + Primary care cases)

    """

    def __init__(
        self,
        latent_severe_primary_rv: RandomVariable,
        latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        severe_admission_obs_process_rv: RandomVariable,
        primary_care_cases_obs_process_rv: RandomVariable,

    ) -> None:  # numpydoc ignore=PR04
        """
        Default constructor

        Parameters
        ----------
        latent_severe_primary_rv
            Latent process for the severe admissions and primary care cases       
        latent_infections_rv
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int_rv
            Generation time (passed to RtInfectionsRenewalModel)
        I0_rv
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process_rv
            Rt process  (passed to RtInfectionsRenewalModel).
        severe_admission_obs_process_rv
            Observation process for the hospital admissions.
        primary_care_cases_obs_process_rv
            Observation process for the primary care cases.

        Returns
        -------
        None
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int_rv=gen_int_rv,
            I0_rv=I0_rv,
            latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=None,  # why is this None?
            Rt_process_rv=Rt_process_rv,
        )

        SeverePrimaryModel.validate(
            latent_severe_primary_rv,
            severe_admission_obs_process_rv, primary_care_cases_obs_process_rv
        )

        self.latent_severe_primary_rv = latent_severe_primary_rv

        if severe_admission_obs_process_rv is None:
            severe_admission_obs_process_rv = NullObservation()
        
        self.severe_admission_obs_process_rv = severe_admission_obs_process_rv

        if primary_care_cases_obs_process_rv is None:
            primary_care_cases_obs_process_rv = NullObservation()

        self.primary_care_cases_obs_process_rv = primary_care_cases_obs_process_rv

    @staticmethod
    def validate(latent_severe_primary_rv,severe_admission_obs_process_rv,primary_care_cases_obs_process_rv) -> None:
        """
        Verifies types and status (RV) of latent and observed hospital admissions

        Parameters
        ----------
        latent_severe_primary_rv
            Latent process for the severe admissions + primary care cases  
        severe_admission_obs_process_rv
            Observation process for the hospital admissions.
        primary_care_cases_obs_process_rv
            Observation process for the primary care cases.            

        Returns
        -------
        None
        """
        assert isinstance(latent_severe_primary_rv, RandomVariable)

        if severe_admission_obs_process_rv is not None:
            assert isinstance(severe_admission_obs_process_rv, RandomVariable)
        if primary_care_cases_obs_process_rv is not None:
            assert isinstance(primary_care_cases_obs_process_rv, RandomVariable)

        return None

    def sample(
        self,
        n_datapoints: int | None = None,
        data_observed_severe_admissions: ArrayLike | None = None,
        data_observed_primary_care_cases: ArrayLike | None = None,
        weekly_obs: bool | None = None,
        padding: int = 0,
        **kwargs,
    ) -> SeverePrimarySample:
        """
        Sample from the SeverePrimary model

        Parameters
        ----------
        n_datapoints
            Number of timepoints to sample (passed to the basic renewal model).
        data_observed_severe_admissions
            The observed severe cases data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        data_observed_primary_care_cases
            The observed primary care cases (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        padding
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        HospModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        
        """
        if data_observed_severe_admissions is None and data_observed_primary_care_cases is None:
        # Case: no data given at all
            if n_datapoints is None:
                raise ValueError(
                    "Either provide both data_observed_hosp_admissions and "
                    "data_observed_genome_copies, or specify n_datapoints."
                )
        # If we reach here, we just use n_datapoints as given
        else:
            # Case: at least one data stream is provided
            if data_observed_severe_admissions is None or data_observed_primary_care_cases is None:
                raise ValueError(
                    "If one data stream is provided, the other must be provided as well."
                )

            if len(data_observed_severe_admissions) != len(data_observed_primary_care_cases):
                raise ValueError(
                    f"Data streams must have the same length, got "
                    f"{len(data_observed_severe_admissions)} and "
                    f"{len(data_observed_primary_care_cases)}."
                )

            if n_datapoints is not None and n_datapoints != len(data_observed_severe_admissions):
                raise ValueError(
                    f"n_datapoints ({n_datapoints}) does not match "
                    f"length of data ({len(data_observed_severe_admissions)})."
                )

            # Infer n_datapoints automatically if not provided
            n_datapoints = len(data_observed_severe_admissions)
            
        if weekly_obs is True:
            n_datapoints = n_datapoints * 7      

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_datapoints=n_datapoints,
            data_observed_infections=None,
            padding=padding,
            **kwargs,
        )
        # Sampling the latent severe cases
        (
            alpha_severe, alpha_primary,
            latent_severe_admissions,latent_primary_care,
            *_,
        ) = self.latent_severe_primary_rv(
            latent_infections=basic_model.latent_infections,
            n = n_datapoints,
            weekly_obs = weekly_obs,
            **kwargs,
        )

        observed_severe_admissions = self.severe_admission_obs_process_rv(
            mu=latent_severe_admissions[-n_datapoints:],
            obs=data_observed_severe_admissions,
            **kwargs,
        )

        observed_primary_care_cases = self.primary_care_cases_obs_process_rv(
            mu=latent_primary_care[-n_datapoints:],
            obs=data_observed_primary_care_cases,
            **kwargs,
        )


        return SeverePrimarySample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            infection_severe_rate=alpha_severe,
            infection_primary_rate = alpha_primary,
            latent_severe_admissions=latent_severe_admissions,
            latent_primary_care_cases = latent_primary_care,
            observed_severe_admissions=observed_severe_admissions,
            observed_primary_care_cases=observed_primary_care_cases,
            
        )