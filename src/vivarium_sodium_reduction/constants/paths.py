from pathlib import Path

import vivarium_sodium_reduction
from vivarium_sodium_reduction.constants import metadata

BASE_DIR = Path(vivarium_sodium_reduction.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/")

