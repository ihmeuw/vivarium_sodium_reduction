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


class SodiumSBPEffect(Component):
    @property
    def name(self):
        return "sodium_sbp_effect"

    def setup(self, builder: Builder):
        self.sodium_exposure = builder.value.get_value("diet_high_in_sodium.exposure")
        self.sodium_exposure_raw = builder.value.get_value("diet_high_in_sodium.raw_exposure")

        builder.value.register_value_modifier(
            "high_systolic_blood_pressure.drop_value",
            modifier=self.sodium_effect_on_sbp_drop,
            requires_columns=["age", "sex"],
            requires_values=[
                "diet_high_in_sodium.exposure",
                "diet_high_in_sodium.raw_exposure",
            ],
        )

    def sodium_effect_on_sbp_drop(self, index, sbp_drop_value):
        sodium_exposure = self.sodium_exposure(index)
        sodium_exposure_raw = self.sodium_exposure_raw(index)

        # FIXME: this should go in the constants.py file
        mmHg_per_g_sodium = (
            5.8 / 6.0
        )  # 5.8 (2.5, 9.2) mmHg decrease per 6g/day sodium decrease

        sodium_drop = sodium_exposure_raw - sodium_exposure

        sbp_drop_due_to_sodium_drop = sodium_drop * mmHg_per_g_sodium

        return sbp_drop_value + sbp_drop_due_to_sodium_drop
