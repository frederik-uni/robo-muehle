from dataclasses import dataclass
from typing import Dict, Any, List

from piecewalker.calibration.helper import parse_calibration_toml
from runtime.args import parse_args

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

@dataclass(frozen=False)
class XYZ:
    x: float
    y: float
    z: float

def list_to_xyz(v: List[float]) -> XYZ:
    if len(v) != 3:
        raise ValueError(f"Expected 3 numbers for XYZ, got {v}")
    return XYZ(float(v[0]), float(v[1]), float(v[2]))

# Scaling Factors for the following indices:
# 0 ---------------------- 1 --------------------- 2
# |                        |                       |
# |                        |                       |
# |         3 ------------ 4 ----------- 5         |
# |         |              |             |         |
# |         |              |             |         |
# |         |      6 ----- 7 ----- 8     |         |
# |         |      |               |     |         |
# |         |      |               |     |         |
# 9 ------  10 --- 11             12 --- 13 ----- 14
# |         |      |               |     |         |
# |         |      |               |     |         |
# |         |      15 ---- 16 ----17     |         |
# |         |              |             |         |
# |         |              |             |         |
# |         18 ----------- 19 ---------- 20        |
# |                        |                       |
# |                        |                       |
# 21 --------------------- 22 -------------------- 23
UV_BY_INDEX = {
    0:(0/6,0/6), 1:(3/6,0/6), 2:(6/6,0/6),
    3:(1/6,1/6), 4:(3/6,1/6), 5:(5/6,1/6),
    6:(2/6,2/6), 7:(3/6,2/6), 8:(4/6,2/6),
    9:(0/6,3/6), 10:(1/6,3/6), 11:(2/6,3/6),
    12:(4/6,3/6), 13:(5/6,3/6), 14:(6/6,3/6),
    15:(2/6,4/6), 16:(3/6,4/6), 17:(4/6,4/6),
    18:(1/6,5/6), 19:(3/6,5/6), 20:(5/6,5/6),
    21:(0/6,6/6), 22:(3/6,6/6), 23:(6/6,6/6),
}

# Affiner Raum: https://de.wikipedia.org/wiki/Affiner_Raum
def get_board_point(idx: int) -> XYZ:
    args = parse_args()
    calibration_file_path: str | None = getattr(args, "calibration_file", None)
    calibration: Dict[str, Any] = parse_calibration_toml(calibration_file_path)

    # Measurement Points from the calibration file as XYZ
    points_xyz: dict[str, XYZ] = {}
    for name, vals in calibration["points"].items():
        points_xyz[name] = list_to_xyz(vals)

    O = points_xyz["corner_a"] # origin (index 0)
    B = points_xyz["corner_b"] # width corner (index 2)
    C = points_xyz["corner_c"] # height corner (index 21)

    # Ratios for this board index
    u, v = UV_BY_INDEX[idx]

    # Width/height vectors in XY only (z intentionally no longer adjusted)
    w_x = B.x - O.x
    w_y = B.y - O.y
    h_x = C.x - O.x
    h_y = C.y - O.y

    # Scale width/height vectors by their respective ratios
    w_x = w_x * u
    w_y = w_y * u
    h_x = h_x * v
    h_y = h_y * v

    x = O.x + w_x + h_x
    y = O.y + w_y + h_y
    z = O.z
    return XYZ(x, y, z)
