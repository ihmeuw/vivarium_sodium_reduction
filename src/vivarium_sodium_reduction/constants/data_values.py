from typing import Dict, NamedTuple, Tuple
from scipy import stats

##############################
# SodiumSBPEffect Parameters #
##############################

class __SodiumSBPEffect(NamedTuple):
    MMHG_PER_G_SODIUM_FOR_LOW_SBP: Tuple[str, stats.norm] = (
        "mmHg_per_g_sodium_for_low_sbp",
        stats.norm(loc=1.0, scale=(1.49 - 0.50)/(2*1.96)),  # UI (.5, 1.49) mmHg from https://doi.org/10.1161/CIRCULATIONAHA.120.050371
    )
    MMHG_PER_G_SODIUM_FOR_HIGH_SBP: Tuple[str, stats.norm] = (
        "mmHg_per_g_sodium_for_high_sbp",
        stats.norm(loc=3.01, scale=(4.02 - 1.99)/(2*1.96)),  # UI (1.99, 4.02) mmHg from https://doi.org/10.1161/CIRCULATIONAHA.120.050371
    )


SodiumSBPEffect = __SodiumSBPEffect()
