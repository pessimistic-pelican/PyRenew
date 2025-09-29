# numpydoc ignore=GL08

from __future__ import annotations

from typing import NamedTuple

from jax.typing import ArrayLike

from pyrenew.deterministic import NullObservation
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel

class AdmissionWastewaterModelSample(NamedTuple):
    """
    A container for holding the output from `model.HospitalAdmissionsModel.sample()`.

    Attributes
    ----------
    Rt : ArrayLike | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : ArrayLike | None, optional
        The estimated number of new infections over time. Defaults to None.
    infection_hosp_rate : ArrayLike | None, optional
        The infected hospitalization rate. Defaults to None.
    latent_hosp_admissions : ArrayLike | None, optional
        The estimated latent hospitalizations. Defaults to None.
    observed_hosp_admissions : ArrayLike | None, optional
        The sampled or observed hospital admissions. Defaults to None.
    alpha
        Time-varying average genome copies per person. Defaults to None.
    latent_genome_copies : ArrayLike | None, optional
        The estimated genome copies per person. Defaults to None.    
    observed_genome_copies : ArrayLike | None, optional
        The sampled or observed genome copies per person. Defaults to None.
    """

    Rt: ArrayLike | None = None
    latent_infections: ArrayLike | None = None
    infection_hosp_rate: ArrayLike | None = None
    latent_hosp_admissions: ArrayLike | None = None
    observed_hosp_admissions: ArrayLike | None = None
    alpha: ArrayLike | None = None
    latent_genome_copies: ArrayLike | None = None
    observed_genome_copies: ArrayLike | None = None

    def __repr__(self):
        return (
            f"HospModelSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"infection_hosp_rate={self.infection_hosp_rate}, "
            f"latent_hosp_admissions={self.latent_hosp_admissions}, "
            f"observed_hosp_admissions={self.observed_hosp_admissions},"
            f"alpha={self.alpha},"
            f"latent_genome_copies={self.observed_hosp_admissions},"
            f"observed_genome_copies={self.observed_hosp_admissions}"
        )
    

class AdmissionsWastewaterModel(Model):
    """
    Hospital Admissions + Wastewater Model (BasicRenewal + HospitalAdmissions + Wastewater)

    This class inherits from pyrenew.models.Model. It extends the
    basic renewal model by adding a hospital admissions module, e.g.,
    pyrenew.latent.HospitalAdmissions, and a wastewater module e.g.,
    pyrenew.latent.Wastewater
    """

    def __init__(
        self,
        latent_hosp_admissions_rv: RandomVariable,
        latent_infections_rv: RandomVariable,
        gen_int_rv: RandomVariable,
        I0_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        hosp_admission_obs_process_rv: RandomVariable,
        latent_genome_copies_rv: RandomVariable,
        genome_copies_obs_process_rv: RandomVariable

    ) -> None:  # numpydoc ignore=PR04
        """
        Default constructor

        Parameters
        ----------
        latent_hosp_admissions_rv
            Latent process for the hospital admissions.
        latent_infections_rv
            The infections latent process (passed to RtInfectionsRenewalModel).
        gen_int_rv
            Generation time (passed to RtInfectionsRenewalModel)
        I0_rv
            Initial infections (passed to RtInfectionsRenewalModel)
        Rt_process_rv
            Rt process  (passed to RtInfectionsRenewalModel).
        hosp_admission_obs_process_rv
            Observation process for the hospital admissions.
        latent_genome_copies_rv
            Latent process for wastewater
        genome_copies_obs_process_rv
            Observation process for the wastewater data

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
        
        AdmissionsWastewaterModel.validate(
            latent_hosp_admissions_rv, hosp_admission_obs_process_rv,
            latent_genome_copies_rv, genome_copies_obs_process_rv
        )

        self.latent_hosp_admissions_rv = latent_hosp_admissions_rv
        self.latent_genome_copies_rv = latent_genome_copies_rv

        if hosp_admission_obs_process_rv is None:
            hosp_admission_obs_process_rv = NullObservation()

        self.hosp_admission_obs_process_rv = hosp_admission_obs_process_rv

        if genome_copies_obs_process_rv is None:
            genome_copies_obs_process_rv = NullObservation()
        
        self.genome_copies_obs_process_rv = genome_copies_obs_process_rv

    @staticmethod
    def validate(latent_hosp_admissions_rv, hosp_admission_obs_process_rv,
                 latent_genome_copies_rv, genome_copies_obs_process_rv) -> None:
        """
        Verifies types and status (RV) of latent and observed hospital admissions

        Parameters
        ----------
        latent_hosp_admissions_rv
            The latent process for the hospital admissions.
        hosp_admission_obs_process_rv
            The observed hospital admissions.

        Returns
        -------
        None
        """
        assert isinstance(latent_hosp_admissions_rv, RandomVariable)
        assert isinstance(latent_genome_copies_rv, RandomVariable)

        if hosp_admission_obs_process_rv is not None:
            assert isinstance(hosp_admission_obs_process_rv, RandomVariable)

        if genome_copies_obs_process_rv is not None:
            assert isinstance(genome_copies_obs_process_rv, RandomVariable)
        return None

    def sample(
        self,
        n_datapoints: int | None = None,
        data_observed_hosp_admissions: ArrayLike | None = None,
        data_observed_genome_copies: ArrayLike | None = None,
        pop: int | float = 5.637e6,
        padding: int = 0,
        **kwargs,
    ) -> AdmissionWastewaterModelSample:
        """
        Sample from the AdmissionWastewater model

        Parameters
        ----------
        n_datapoints
            Number of timepoints to sample (passed to the basic renewal model).
        data_observed_hosp_admissions
            The observed hospitalization data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        data_observed_genome_copies
            The observed genome copies data (passed to the basic renewal
            model). Defaults to None (simulation, rather than fit).
        padding
            Number of padding timepoints to add to the beginning of the
            simulation. Defaults to 0.
        **kwargs
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        AdmissionWastewaterModelSample

        See Also
        --------
        basic_renewal.sample : For sampling the basic renewal model
        sample_observed_admissions : For sampling observed hospital admissions
        """
        if data_observed_hosp_admissions is None and data_observed_genome_copies is None:
        # Case: no data given at all
            if n_datapoints is None:
                raise ValueError(
                    "Either provide both data_observed_hosp_admissions and "
                    "data_observed_genome_copies, or specify n_datapoints."
                )
        # If we reach here, we just use n_datapoints as given
        else:
            # Case: at least one data stream is provided
            if data_observed_hosp_admissions is None or data_observed_genome_copies is None:
                raise ValueError(
                    "If one data stream is provided, the other must be provided as well."
                )

            if len(data_observed_hosp_admissions) != len(data_observed_genome_copies):
                raise ValueError(
                    f"Data streams must have the same length, got "
                    f"{len(data_observed_hosp_admissions)} and "
                    f"{len(data_observed_genome_copies)}."
                )

            if n_datapoints is not None and n_datapoints != len(data_observed_hosp_admissions):
                raise ValueError(
                    f"n_datapoints ({n_datapoints}) does not match "
                    f"length of data ({len(data_observed_hosp_admissions)})."
                )

            # Infer n_datapoints automatically if not provided
            n_datapoints = len(data_observed_hosp_admissions)

        # Getting the initial quantities from the basic model
        basic_model = self.basic_renewal.sample(
            n_datapoints=n_datapoints,
            data_observed_infections=None,
            padding=padding,
            **kwargs,
        )
        # Sampling the latent hospital admissions
        (
            infection_hosp_rate,
            latent_hosp_admissions,
            *_,
        ) = self.latent_hosp_admissions_rv(
            latent_infections=basic_model.latent_infections,
            n = n_datapoints,
            **kwargs,
        )
        observed_hosp_admissions = self.hosp_admission_obs_process_rv(
            mu=latent_hosp_admissions[-n_datapoints:],
            obs=data_observed_hosp_admissions,
            **kwargs,
        )

        #Sample the latent wastewater
        (
            alpha,
            latent_genome_copies
        ) = self.latent_genome_copies_rv(
            latent_infections=basic_model.latent_infections,
            n = n_datapoints,
            **kwargs,
        )

        observed_genome_copies = self.genome_copies_obs_process_rv(
            mu = latent_genome_copies[-n_datapoints:] / pop,
            obs = data_observed_genome_copies,
            **kwargs,
        )

        return AdmissionWastewaterModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            infection_hosp_rate=infection_hosp_rate,
            latent_hosp_admissions=latent_hosp_admissions,
            observed_hosp_admissions=observed_hosp_admissions,
            alpha = alpha,
            latent_genome_copies=latent_genome_copies,
            observed_genome_copies=observed_genome_copies,
        )