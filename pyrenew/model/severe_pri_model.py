import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike

import datetime as dt
from functools import partial


import pandas as pd
from pandas import DataFrame

import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

from pyrenew import observation, randomvariable, model
import pyrenew.transformation as transformation
from pyrenew.arrayutils import tile_until_n
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    Infections,
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import daily_to_mmwr_epiweekly



class LatentInfectionProcess(RandomVariable):
    def __init__(
        self,
        i0_first_obs_n_rv: RandomVariable,
        Rt_process_rv: RandomVariable,
        feedback: bool,
        generation_interval_pmf_rv: RandomVariable,       # GI
        infection_feedback_strength_rv: RandomVariable,   # depletion strength
        infection_feedback_pmf_rv: RandomVariable,        # depletion pmf
        n_initialization_points: int,                     # days before first observation
        n_newton_steps=4,
    ) -> None:
        
        self.feedback = feedback
        if self.feedback:
            self.inf_with_feedback_proc = InfectionsWithFeedback(
                infection_feedback_strength=infection_feedback_strength_rv,
                infection_feedback_pmf=infection_feedback_pmf_rv,
            )
        else:
            self.inf_with_no_feedback_proc = Infections()
        
        self.i0_first_obs_n_rv = i0_first_obs_n_rv
        self.Rt_process_rv = Rt_process_rv

        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.n_initialization_points = n_initialization_points
        self.n_newton_steps = n_newton_steps

    def validate(self):
        pass

    def sample(self,
               n_days_post_init: int,
               **kwargs,):
        """
        Sample latent infections.

        Parameters
        ----------
        n_days_post_init
            Number of days of infections to sample, not including
            the initialization period.
        """

        # === 1. Draw parameters ===
        gi_pmf   = self.generation_interval_pmf_rv()    

        Rt_daily = self.Rt_process_rv(
            n = n_days_post_init
        )

        # === . Convert weekly Rt to exponential growth rate 'r' (Newton solver) ===
        r_approx = partial(
                r_approx_from_R,
                g=gi_pmf,
                n_newton_steps=self.n_newton_steps
                )

        initial_exp_growth_rate = r_approx(Rt_daily[0])


        # === 5. Initialize infections prior to first observation ===
        infection_initialization_process = InfectionInitializationProcess(
            'I0_initialisation',
            I_pre_init_rv=self.i0_first_obs_n_rv,
            infection_init_method= InitializeInfectionsExponentialGrowth(
                self.n_initialization_points,
                DeterministicVariable("initial_r", initial_exp_growth_rate)
            )
        )
        
        i0 = infection_initialization_process()

        if self.feedback:
            # === 6. Renewal equation with feedback ===
            infections= self.inf_with_feedback_proc(
                Rt=Rt_daily[:, None],   # shape (T,1)
                I0=i0[:, None],         # shape (T0,1)
                gen_int=gi_pmf,
            )

            post_init_infections = infections.post_initialization_infections[:,0]

            numpyro.deterministic("Adj_Rt_daily", infections.rt)   

        else:
            infections = self.inf_with_no_feedback_proc(
                Rt = Rt_daily,
                I0=i0,
                gen_int = gi_pmf
            )
            post_init_infections = infections.post_initialization_infections

        latent_inf = jnp.concatenate([
            i0,
            post_init_infections
        ])

        numpyro.deterministic("Rt_daily", Rt_daily)     
        numpyro.deterministic("latent_infections", latent_inf)

        return latent_inf
    

class PolyVisitObservationProcess(RandomVariable):
    def __init__(
        self,
        p_poly_mean_rv: RandomVariable,
        inf_to_poly_rv: RandomVariable,
        poly_obs_process_rv: RandomVariable,
        p_sym: RandomVariable,
    ) -> None:
        self.p_poly_mean_rv = p_poly_mean_rv
        self.inf_to_poly_rv = inf_to_poly_rv
        self.poly_obs_process_rv = poly_obs_process_rv
        self.p_sym = p_sym

    def validate(self):
        pass

    def sample(
        self,
        n_datapoints: int,
        latent_infections: ArrayLike,
        population_size: int,
        data_observed: ArrayLike,
        **kwargs
    ) -> tuple[ArrayLike]:
        """
        Observe and/or predict poly visit values
        """
        inf_to_poly = self.inf_to_poly_rv()

        potential_latent_poly_visits, poly_visit_offset = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=latent_infections,
                delay_incidence_to_observation_pmf=inf_to_poly,
            )
        )

        # model_t_first_latent_poly_visit = poly_visit_offset + model_t_first_latent_infection
        p_sym = self.p_sym()

        p_poly_mean = self.p_poly_mean_rv(p_sym=p_sym)

        # daily expected poly visits
        mu_poly_daily = potential_latent_poly_visits * p_poly_mean * population_size
        mu_poly_daily = mu_poly_daily[-(n_datapoints):]

        numpyro.deterministic("mu_poly_daily", mu_poly_daily)
        numpyro.deterministic("p_poly", p_poly_mean)

        # 4. Aggregate daily -> weekly
        n_days = mu_poly_daily.shape[0]
        n_weeks = n_days // 7
        mu_weekly = mu_poly_daily.reshape(n_weeks, 7).sum(axis=1)

        numpyro.deterministic("mu_poly_weekly", mu_weekly)

        observed_poly_visits = self.poly_obs_process_rv(
            mu=mu_weekly,
            obs=data_observed,
        )

        return observed_poly_visits, mu_weekly
    

class HospObservationProcess(RandomVariable):
    def __init__(
        self,
        p_hosp_mean_rv: RandomVariable,
        inf_to_hosp_rv: RandomVariable,
        hosp_obs_process_rv: RandomVariable,
    ) -> None:
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.hosp_obs_process_rv = hosp_obs_process_rv

    def validate(self):
        pass

    def sample(
        self,
        n_datapoints: int,
        latent_infections: ArrayLike,
        population_size: int,
        data_observed: ArrayLike,
        **kwargs
    ) -> tuple[ArrayLike]:
        """
        Observe and/or predict hospitalisation
        """
        inf_to_hosp = self.inf_to_hosp_rv()

        potential_latent_hosp, hosp_offset = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=latent_infections,
                delay_incidence_to_observation_pmf=inf_to_hosp,
            )
        )

        p_hosp_mean = self.p_hosp_mean_rv()

        # daily expected poly visits
        mu_hosp_daily = potential_latent_hosp * p_hosp_mean * population_size
        mu_hosp_daily = mu_hosp_daily[-(n_datapoints):]

        numpyro.deterministic("mu_hosp_daily", mu_hosp_daily)
        numpyro.deterministic("p_hosp", p_hosp_mean)

        # 4. Aggregate daily -> weekly
        n_days = mu_hosp_daily.shape[0]
        n_weeks = n_days // 7
        mu_weekly = mu_hosp_daily.reshape(n_weeks, 7).sum(axis=1)

        numpyro.deterministic("mu_hosp_weekly", mu_weekly)

        observed_hosp = self.hosp_obs_process_rv(
            mu=mu_weekly,
            obs=data_observed,
        )

        return observed_hosp, mu_weekly
    

class PyrenewModel(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        population_size: int,
        latent_infection_process_rv: LatentInfectionProcess,
        poly_visit_obs_process_rv: PolyVisitObservationProcess,
        hosp_admit_obs_process_rv: HospObservationProcess,
        # wastewater_obs_process_rv: WastewaterObservationProcess,
    ) -> None:  # numpydoc ignore=GL08
        self.population_size = population_size
        self.latent_infection_process_rv = latent_infection_process_rv
        self.poly_visit_obs_process_rv = poly_visit_obs_process_rv
        self.hosp_admit_obs_process_rv = hosp_admit_obs_process_rv
        # self.wastewater_obs_process_rv = wastewater_obs_process_rv

    def validate(self) -> None:  # numpydoc ignore=GL08
        pass

    def sample(
        self,
        n_datapoints: int | None = None,
        data_observed_poly: ArrayLike | None = None,
        data_observed_hosp: ArrayLike | None = None,
        sample_poly_visits: bool = True,
        sample_hospital_admissions: bool = True,
        **kwargs,
        # sample_wastewater: bool = False,
    ) -> dict[str, ArrayLike]:  # numpydoc ignore=GL08
        
        if data_observed_poly is None and data_observed_hosp is None:
            n_days_post_init = n_datapoints
        else:
            assert len(data_observed_poly) == len(data_observed_hosp)
            n_days_post_init = len(data_observed_poly) * 7

        latent_infections = self.latent_infection_process_rv(
            n_days_post_init=n_days_post_init,
        )
        # first_latent_infection_dow = (
        #     data['first_data_date_overall'] - dt.timedelta(days=n_init_days)
        # ).weekday()

        # observed_poly_visits = None
        # mu_poly_weekly = None
        # observed_admissions = None
        # site_level_observed_wastewater = None
        # population_level_latent_wastewater = None

        if sample_poly_visits:
            observed_poly_visits, mu_poly_weekly = self.poly_visit_obs_process_rv(
                n_datapoints = n_days_post_init,
                latent_infections=latent_infections,
                population_size=self.population_size,
                data_observed=data_observed_poly,
                **kwargs
            )

        if sample_hospital_admissions:
            observed_admissions, _ = self.hosp_admit_obs_process_rv(
                n_datapoints = n_days_post_init,
                latent_infections=latent_infections,
                population_size=self.population_size,
                data_observed=data_observed_hosp,
                **kwargs
            )
        
        return {
            "poly_visits": observed_poly_visits,
            "hospital_admissions": observed_admissions,
            # "site_level_wastewater_conc": site_level_observed_wastewater,
            # "population_level_latent_wastewater_conc": population_level_latent_wastewater,
        }