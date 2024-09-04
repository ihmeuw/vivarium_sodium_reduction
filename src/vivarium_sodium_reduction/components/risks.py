from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
from gbd_mapping import risk_factors
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import get_hash
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import Risk
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.utilities import EntityString


class DropValueRisk(Risk):
    def __init__(self, risk: str):
        super().__init__(risk)
        self.raw_exposure_pipeline_name = f"{self.risk.name}.raw_exposure"
        self.drop_value_pipeline_name = f"{self.risk.name}.drop_value"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.raw_exposure = self.get_raw_exposure_pipeline(builder)
        self.drop_value = self.get_drop_value_pipeline(builder)

    def get_drop_value_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.drop_value_pipeline_name,
            source=lambda index: pd.Series(0.0, index=index),
        )

    def get_raw_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.raw_exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[self.propensity_pipeline_name],
        )

    def get_exposure_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.get_current_exposure,
            requires_columns=["age", "sex"],
            requires_values=[self.propensity_pipeline_name],
            preferred_post_processor=self.get_drop_value_post_processor(builder, self.risk),
        )

    def get_drop_value_post_processor(self, builder: Builder, risk: EntityString):
        drop_value_pipeline = builder.value.get_value(self.drop_value_pipeline_name)

        def post_processor(exposure, _):
            drop_values = drop_value_pipeline(exposure.index)
            return exposure - drop_values

        return post_processor


class CorrelatedRisk(DropValueRisk):
    """A risk that can be correlated with another risk.

    TODO: document strategy used in this component in more detail,
    Abie had an AI adapt it from https://github.com/ihmeuw/vivarium_nih_us_cvd"""

    @property
    def columns_created(self) -> List[str]:
        return []

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [self.propensity_column_name]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [],
        }

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pass

    def on_time_step_prepare(self, event: Event) -> None:
        pass


class ThresholdRisk(Component):
    """A component that generates a risk based on a threshold of another risk."""

    def __init__(self, risk: str, threshold: str):
        super().__init__()
        self.risk = EntityString(risk)
        self.threshold = float(threshold)
        self.exposure_pipeline_name = f"{self.risk.name}.threshold_exposure"

    def setup(self, builder: Builder) -> None:
        self.continuous_exposure = builder.value.get_value(self.risk.exposure_pipeline_name)
        self.threshold_exposure = builder.value.register_value_producer(
            self.exposure_pipeline_name,
            source=self.continuous_exposure
        )

    def get_exposure_threshold_value(self, value):

        return np.where(value <= self.threshold, 'cat1', 'cat2')



class RiskCorrelation(Component):
    """A component that generates a specified correlation between two risk exposures."""

    @property
    def columns_created(self) -> List[str]:
        return self.propensity_column_names + self.exposure_column_names

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {"requires_columns": ["age"] + self.ensemble_propensities}

    def __init__(self, risk1: str, risk2: str, correlation: str):
        super().__init__()
        correlated_risks = [risk1, risk2]
        correlation_matrix = np.array([[1, float(correlation)], [float(correlation), 1]])
        self.correlated_risks = [EntityString(risk) for risk in correlated_risks]
        self.correlation_matrix = correlation_matrix
        self.propensity_column_names = [
            f"{risk.name}_propensity" for risk in self.correlated_risks
        ]
        self.exposure_column_names = [
            f"{risk.name}_exposure" for risk in self.correlated_risks
        ]
        self.ensemble_propensities = [
            f"ensemble_propensity_" + risk
            for risk in self.correlated_risks
            if risk_factors[risk.name].distribution == "ensemble"
        ]

    def setup(self, builder: Builder) -> None:
        self.distributions = {
            risk: builder.components.get_component(risk).exposure_distribution
            for risk in self.correlated_risks
        }
        self.exposures = {
            risk: builder.value.get_value(f"{risk.name}.exposure")
            for risk in self.correlated_risks
        }
        self.input_draw = builder.configuration.input_data.input_draw_number
        self.random_seed = builder.configuration.randomness.random_seed

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["age"]).get(pop_data.index)
        propensities = pd.DataFrame(index=pop.index)

        np.random.seed(get_hash(f"{self.input_draw}_{self.random_seed}"))
        probit_propensity = np.random.multivariate_normal(
            mean=[0] * len(self.correlated_risks), cov=self.correlation_matrix, size=len(pop)
        )
        correlated_propensities = scipy.stats.norm().cdf(probit_propensity)
        propensities[self.propensity_column_names] = correlated_propensities

        def get_exposure_from_propensity(propensity_col: pd.Series) -> pd.Series:
            risk = propensity_col.name.replace("_propensity", "")
            exposure_values = self.distributions["risk_factor." + risk].ppf(propensity_col)
            return pd.Series(exposure_values)

        exposures = propensities.apply(get_exposure_from_propensity)
        exposures.columns = [
            col.replace("_propensity", "_exposure") for col in propensities.columns
        ]

        self.population_view.update(pd.concat([propensities, exposures], axis=1))

    def on_time_step_prepare(self, event: Event) -> None:
        for risk in self.exposures:
            exposure_values = self.exposures[risk](event.index)
            exposure_col = pd.Series(exposure_values, name=f"{risk.name}_exposure")
            self.population_view.update(exposure_col)


class SodiumSBPEffect(Component):
    @property
    def name(self):
        return "sodium_sbp_effect"

    def setup(self, builder: Builder):
        self.sodium_exposure = builder.value.get_value("diet_high_in_sodium.exposure")
        self.sodium_exposure_raw = builder.value.get_value("diet_high_in_sodium.raw_exposure")

        builder.value.register_value_modifier(
            "high_systolic_blood_pressure.drop_value",
            modifier=self.sodium_effect_on_sbp,
            requires_columns=["age", "sex"],
            requires_values=[
                "diet_high_in_sodium.exposure",
                "diet_high_in_sodium.raw_exposure",
            ],
        )

    def sodium_effect_on_sbp(self, index, sbp_drop_value):
        sodium_exposure = self.sodium_exposure(index)
        sodium_exposure_raw = self.sodium_exposure_raw(index)

        mmHg_per_g_sodium = 5.8/6.0  # 5.8 (2.5, 9.2) mmHg decrease per 6g/day sodium decrease

        sbp_increase = pd.Series(0, index=index)
        sodium_drop = sodium_exposure_raw - sodium_exposure

        sbp_drop_due_to_sodium_drop = sodium_drop * mmHg_per_g_sodium

        return sbp_drop_value + sbp_drop_due_to_sodium_drop
