import tomllib
from pathlib import Path
from typing import Any

from piecewalker.paths import Paths, build_paths

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

_CFG: dict[str, Any] | None = None

def get_config() -> dict[str, Any]:
    global _CFG
    if _CFG is not None:
        return _CFG

    paths: Paths = build_paths()
    cfg_path: Path = paths.niryo_cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found at path: {cfg_path}")

    with cfg_path.open("rb") as f:
        data = tomllib.load(f)

    piecewalker = data.get("piecewalker", {})

    _CFG = {
        "path": cfg_path,
        "general": data.get("general", {}),
        "piecewalker": {
            "robot_ip": str(piecewalker.get("robot_ip", "169.254.200.200")),

            "chip_height": float(piecewalker.get("chip_height", 0.0045)),
            "fixed_z": float(piecewalker.get("fixed_z", 0.0895)),

            "pick_z_offset": float(piecewalker.get("pick_z_offset", 0.0)),
            "place_z_offset": float(piecewalker.get("place_z_offset", 0.0)),

            # nested tables (dicts)
            "idle_pose": piecewalker.get("idle_pose", {}),
            "grasp_params": piecewalker.get("grasp_params", {}),
        },
    }
    return _CFG