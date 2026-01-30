from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import fmean

from piecewalker.calibration.helper import CALIBRATION_FILE_PREFIX, CALIBRATION_INDEX_WIDTH, CALIBRATION_FILE_EXTENSION
from piecewalker.calibration.helper import get_first_free_cfg_idx
from piecewalker import ned2_provider
from piecewalker.config import get_config
from piecewalker.paths import build_paths
from runtime.args import parse_args

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

class CalibrationStep(Enum):
    CORNER_A = "Corner A (origin)"
    CORNER_B = "Corner B (width)"
    CORNER_C = "Corner C (height)"
    UNPLACED_CHIPS = "Unplaced Chips Stack"
    REMOVED_CHIPS = "Removed Chips Stack"

@dataclass(frozen=True)
class XYZ:
    x: float
    y: float
    z: float

def _record_xyz() -> XYZ:
    """
    Returns the end effector pose for a particular calibration point.
    Documentation: https://niryorobotics.github.io/pyniryo/v1.2.1-1/examples/examples_movement.html#pose
    """
    ned2 = ned2_provider.get_ned2()

    pose = ned2.robot.get_pose()

    return XYZ(pose.x, pose.y, pose.z)


# noinspection PyListCreation
def _build_toml_content(points: dict[CalibrationStep, XYZ]) -> str:
    args = parse_args()
    fixed_z: bool = getattr(args, "fixed_z", False)
    lines: list[str] = []
    lines.append("[general]")
    lines.append(f"used_fixed_z = {str(fixed_z).lower()}")
    lines.append("")

    lines.append("[points]")
    for step, p in points.items():
        # Use enum name as key
        key = step.name.lower()
        lines.append(f'{key} = [{p.x:.6f}, {p.y:.6f}, {p.z:.6f}]')

    lines.append("")
    return "\n".join(lines)

def _run_calibration() -> str:
    args = parse_args()
    cfg = get_config()

    # Record actual x,y,z coordinates for all points
    points: dict[CalibrationStep, XYZ] = {}
    for step in CalibrationStep:
        input(f"Position robot to the: \"{step.value}\", then press Enter to record...")
        points[step] = _record_xyz()

    # Average the board points, or override them with the fixed_z value
    fixed_z: bool = getattr(args, "fixed_z", False)
    board_z_val: float
    if fixed_z:
        board_z_val: float | None = float(cfg["piecewalker"]["fixed_z"])
    else:
        board_z_val = fmean([
            points[CalibrationStep.CORNER_A].z,
            points[CalibrationStep.CORNER_B].z,
            points[CalibrationStep.CORNER_C].z,
        ])
    for step in (CalibrationStep.CORNER_A, CalibrationStep.CORNER_B, CalibrationStep.CORNER_C):
        p = points[step]
        points[step] = XYZ(p.x, p.y, board_z_val)

    return _build_toml_content(points)

def _get_next_file_name() -> Path:
    """
    finds the first free index in the calibrations dir
    and then returns the file name as a Path object
    """
    next_idx = get_first_free_cfg_idx()
    file_name = Path(f"{CALIBRATION_FILE_PREFIX}{next_idx:0{CALIBRATION_INDEX_WIDTH}}{CALIBRATION_FILE_EXTENSION}")

    return file_name

def _gen_calibration_file() -> Path:
    """
    generates a new calibration file.
    """
    paths = build_paths()
    paths.calibrations_dir.mkdir(parents=True, exist_ok=True)

    file_name = _get_next_file_name()
    target_path = paths.calibrations_dir / file_name

    content = _run_calibration()
    target_path.write_text(content, encoding="utf-8")
    return target_path

if __name__ == "__main__":
    # I just like to calibrate and initialize the robot at the start of the script
    # so that it's calibrated and moved into position initially...
    _ = ned2_provider.get_ned2()

    out_path = _gen_calibration_file()

    ned2_provider.close_ned2()
    print("Wrote:", out_path)
