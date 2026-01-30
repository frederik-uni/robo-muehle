from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

# (I know this is somewhat over-engineered for our current logic,
# the Paths used to be dynamic and I don't wanna mess with it anymore...)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

_PATHS: Paths | None = None

@dataclass(frozen=True)
class Paths:
    project_root: Path
    configs_dir: Path
    calibrations_dir: Path
    niryo_cfg_path: Path

def build_paths() -> Paths:
    global _PATHS
    if not _PATHS is None:
        return _PATHS

    paths = Paths(
        project_root=PROJECT_ROOT,
        configs_dir=CONFIGS_DIR,
        calibrations_dir=CONFIGS_DIR / "calibrations",
        niryo_cfg_path=CONFIGS_DIR / "niryo_config.toml",
    )

    _PATHS = paths
    return _PATHS