import pandas as pd
from typing import Dict, Any
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

class RelativeShiftIntervention(Component):
    """Applies a relative shift to a target value."""

    CONFIGURATION_DEFAULTS = {
        'intervention': {
            'shift_factor': 0.1,
            'age_start': 0,
            'age_end': 125,
        }
    }

    def __init__(self, target: str):
        super().__init__()
        self.target = target

    @property
    def name(self) -> str:
        return f"relative_shift_intervention.{self.target}"

    @property
    def configuration_defaults(self) -> Dict[str, Dict[str, Any]]:
        return {
            f'{self.name}': self.CONFIGURATION_DEFAULTS['intervention']
        }

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration[self.name]
        self.shift_factor = self.config.shift_factor
        self.age_start = self.config.age_start
        self.age_end = self.config.age_end

        self.population_view = builder.population.get_view(['age'])

        builder.value.register_value_modifier(
            self.target,
            modifier=self.adjust_exposure,
            requires_columns=['age']
        )

    def adjust_exposure(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        pop = self.population_view.get(index)
        applicable_index = pop.loc[
            (self.age_start <= pop.age) & (pop.age < self.age_end)
        ].index
        exposure.loc[applicable_index] *= self.shift_factor
        return exposure