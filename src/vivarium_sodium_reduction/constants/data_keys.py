from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __IschemicHeartDisease(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.ischemic_heart_disease.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.ischemic_heart_disease.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.ischemic_heart_disease.disability_weight")
    EMR: TargetString = TargetString("cause.ischemic_heart_disease.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.ischemic_heart_disease.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.ischemic_heart_disease.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "ischemic_heart_disease"

    @property
    def log_name(self):
        return "ischemic_heart_disease"


IHD = __IschemicHeartDisease()


class __IschemicStroke(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.ischemic_stroke.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.ischemic_stroke.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.ischemic_stroke.disability_weight")
    EMR: TargetString = TargetString("cause.ischemic_stroke.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.ischemic_stroke.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.ischemic_stroke.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "ischemic_stroke"

    @property
    def log_name(self):
        return "ischemic_stroke"


ISCHEMIC_STROKE = __IschemicStroke()


class __StomachCancer(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.stomach_cancer.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.stomach_cancer.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.stomach_cancer.disability_weight")
    EMR: TargetString = TargetString("cause.stomach_cancer.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.stomach_cancer.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.stomach_cancer.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "stomach_cancer"

    @property
    def log_name(self):
        return "stomach_cancer"


STOMACH_CANCER = __StomachCancer()


class __IntracerebralHemorrhage(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.intracerebral_hemorrhage.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.intracerebral_hemorrhage.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.intracerebral_hemorrhage.disability_weight")
    EMR: TargetString = TargetString("cause.intracerebral_hemorrhage.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.intracerebral_hemorrhage.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.intracerebral_hemorrhage.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "intracerebral_hemorrhage"

    @property
    def log_name(self):
        return "intracerebral_hemorrhage"


INTRACEREBRAL_HEMORRHAGE = __IntracerebralHemorrhage()


class __SubarachnoidHemorrhage(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString("cause.subarachnoid_hemorrhage.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.subarachnoid_hemorrhage.incidence_rate")
    DISABILITY_WEIGHT: TargetString = TargetString("cause.subarachnoid_hemorrhage.disability_weight")
    EMR: TargetString = TargetString("cause.subarachnoid_hemorrhage.excess_mortality_rate")
    CSMR: TargetString = TargetString("cause.subarachnoid_hemorrhage.cause_specific_mortality_rate")
    RESTRICTIONS: TargetString = TargetString("cause.subarachnoid_hemorrhage.restrictions")

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return "subarachnoid_hemorrhage"

    @property
    def log_name(self):
        return "subarachnoid_hemorrhage"


SUBARACHNOID_HEMORRHAGE = __SubarachnoidHemorrhage()


class __HighSBP(NamedTuple):
    DISTRIBUTION: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.distribution"
    )
    EXPOSURE_MEAN: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.exposure"
    )
    EXPOSURE_SD: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.exposure_standard_deviation"
    )
    EXPOSURE_WEIGHTS: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.exposure_distribution_weights"
    )
    RELATIVE_RISK: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.relative_risk"
    )
    PAF: TargetString = TargetString(
        "risk_factor.high_systolic_blood_pressure.population_attributable_fraction"
    )
    TMRED: TargetString = TargetString("risk_factor.high_systolic_blood_pressure.tmred")

    @property
    def name(self):
        return "high_sbp"

    @property
    def log_name(self):
        return self.name.replace("_", " ")


SBP = __HighSBP()


class __HighSodium(NamedTuple):
    DISTRIBUTION: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.distribution"
    )
    EXPOSURE_MEAN: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.exposure"
    )
    EXPOSURE_SD: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.exposure_standard_deviation"
    )
    EXPOSURE_WEIGHTS: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.exposure_distribution_weights"
    )
    RELATIVE_RISK: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.relative_risk"
    )
    PAF: TargetString = TargetString(
        "risk_factor.diet_high_in_sodium.population_attributable_fraction"
    )
    TMRED: TargetString = TargetString("risk_factor.diet_high_in_sodium.tmred")

    @property
    def name(self):
        return "high_sodium"

    @property
    def log_name(self):
        return self.name.replace("_", " ")


SODIUM = __HighSodium()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    IHD,
    ISCHEMIC_STROKE,
    STOMACH_CANCER,
    INTRACEREBRAL_HEMORRHAGE,
    SUBARACHNOID_HEMORRHAGE,
    SBP,
    SODIUM,
]
