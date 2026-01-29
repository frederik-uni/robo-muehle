from pathlib import Path
from typing import Dict, Any

from pyniryo import NiryoRobot, PoseObject

from piecewalker.board_model import XYZ, get_board_point, list_to_xyz
from piecewalker.calibration.helper import parse_calibration_toml
from piecewalker.config import get_config
from runtime.args import parse_args

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

class Ned2:
    def _build_pose_from_xyz(self, xyz: XYZ) -> PoseObject:
        gp = self.cfg["piecewalker"]["grasp_params"]
        roll = float(gp["roll"])
        pitch = float(gp["pitch"])
        yaw = float(gp["yaw"])

        return PoseObject(xyz.x, xyz.y, xyz.z, roll, pitch, yaw)

    def _pick_z(self, z: float) -> float:
        return z + self.pick_z_offset

    def _place_z(self, z: float) -> float:
        return z + self.place_z_offset

    def _pick_from_above(self, pick_position: PoseObject) -> None:
        """
        Picks up a piece from the board by first hovering above the pick position,
        picking up the piece and then retreating back up.
        :param pick_position: PoseObject describing the coordinates where to pick up the piece from.
        """
        # go above the pick position (same x/y as from_pose, safe z = idle z)
        from_above = PoseObject(
            pick_position.x, pick_position.y, self._idle_pose.z,
            pick_position.roll, pick_position.pitch, pick_position.yaw
        )
        print("\nMoving above pick position (from_above):\n", from_above)
        self.robot.move_pose(from_above)

        # Adjust the pick_position to use a small z_offset
        # (Don't change the passed object to avoid side effects)
        adjusted_pick_pose = PoseObject(
            pick_position.x, pick_position.y, self._pick_z(pick_position.z),
            pick_position.roll, pick_position.pitch, pick_position.yaw
        )

        print("\nDescending to pick (adjusted_pick_pose):\n", adjusted_pick_pose)
        self.robot.move_pose(adjusted_pick_pose)
        self.robot.grasp_with_tool()

        # retreat back up
        print("\nRetreating up (from_above):\n", from_above)
        self.robot.move_pose(from_above)

    def _place_from_above(self, place_position: PoseObject) -> None:
        """
        Places a piece onto the board by first hovering above the place position,
        placing the piece and then retreating back up.
        :param place_position: PoseObject describing the coordinates where to place the piece.
        """

        # go above the place position
        to_above = PoseObject(
            place_position.x, place_position.y, self._idle_pose.z,
            place_position.roll, place_position.pitch, place_position.yaw
        )
        print("\nMoving above place position (to_above):\n", to_above)
        self.robot.move_pose(to_above)

        # Adjust the place_pose to use a small z_offset
        # (Don't change the passed object to avoid side effects)
        adjusted_place_pose = PoseObject(
            place_position.x, place_position.y, self._place_z(place_position.z),
            place_position.roll, place_position.pitch, place_position.yaw
        )

        print("\nDescending to place (adjusted_place_pose):\n", adjusted_place_pose)
        self.robot.move_pose(adjusted_place_pose)
        self.robot.release_with_tool()

        # retreat back up
        print("\nRetreating up (to_above):\n", to_above)
        self.robot.move_pose(to_above)

    def move(self, *, from_idx: int | None, to_idx: int | None, back_to_idle: bool = True) -> None:
        """
        Pick-and-place one piece using a safe "hover -> descend -> retreat" sequence.

        Sequence:
          1) Pick:  move above the from_idx -> descend -> grasp -> retreat above.
          2) Place: move above the to_idx -> descend -> grasp -> retreat above.
          3) Optional: if back_to_idle=True, move back to the configured idle pose (niryo_config.toml).

        from_idx:
          - int (0..23): pick from board index
          - None: pick from calibration["points"]["unplaced_chips"] (uses stack height, decrements unplaced count)

        to_idx:
          - int (0..23): place to board index
          - None: place to calibration["points"]["removed_chips"] (uses stack height, increments removed count)

        back_to_idle:
          - True: return to idle pose after placing a piece
          - False: stop after the place "retreat up" (useful for chaining moves)
        """
        if from_idx is not None and not (0 <= from_idx <= 23):
            raise ValueError(f"from_idx out of range: {from_idx}")
        if to_idx is not None and not (0 <= to_idx <= 23):
            raise ValueError(f"to_idx out of range: {to_idx}")

        from_is_stack = (from_idx is None)
        to_is_stack = (to_idx is None)

        from_pose: PoseObject
        to_pose: PoseObject

        # Move From "unplaced_chips" Stack, if no from_idx was provided
        if from_idx is None:
            if self.unplaced_chips_count <= 0:
                raise ValueError("No unplaced chips left to pick from (unplaced_chips_count <= 0).")

            unplaced_chips_xyz: XYZ = list_to_xyz(self.calibration["points"]["unplaced_chips"])
            unplaced_chips_xyz.z = self._stack_z(unplaced_chips_xyz.z, self.unplaced_chips_count, for_pick=True)
            from_pose = self._build_pose_from_xyz(unplaced_chips_xyz)
        else:
            # Move to concrete board position, if from_idx was provided
            from_xyz: XYZ = get_board_point(from_idx)
            from_pose = self._build_pose_from_xyz(from_xyz)

        # Move To "removed_chips" Stack, if no to_idx was provided
        if to_idx is None:
            if self.removed_chips_count >= 9:
                raise ValueError("Removed chips stack is already at max height (removed_chips_count >= 9).")

            removed_chips_xyz: XYZ = list_to_xyz(self.calibration["points"]["removed_chips"])
            removed_chips_xyz.z = self._stack_z(removed_chips_xyz.z, self.removed_chips_count, for_pick=False)
            to_pose = self._build_pose_from_xyz(removed_chips_xyz)
        else:
            # Move to concrete board position, if to_idx was provided
            to_xyz: XYZ = get_board_point(to_idx)
            to_pose = self._build_pose_from_xyz(to_xyz)

        # Movement
        # Ensure the tool is in a released state
        self.robot.release_with_tool()

        # ---------- PICK ----------
        self._pick_from_above(from_pose)
        # Decrement unplaced_chips_count, if picked from it
        if from_is_stack:
            self.unplaced_chips_count -= 1  # commit only after successful pick

        # ---------- PLACE ----------
        self._place_from_above(to_pose)
        # Increment removed_chips_count, if placed onto it
        if to_is_stack:
            self.removed_chips_count += 1  # commit only after successful place

        # ---- (Optionally) BACK TO IDLE ----
        if back_to_idle:
            print("\nBack into idle pose:\n", self._idle_pose)
            self.robot.move_pose(self._idle_pose)

    def close(self) -> bool:
        if self.robot is not None:
            self.robot.close_connection()
            self.robot = None
            return True
        return False

    def _stack_z(self, base_z: float, count: int, for_pick: bool) -> float:
        """
        Returns the z-value adjusted for stack height.
        """
        chip_h = float(self.cfg["piecewalker"]["chip_height"])

        if count < 0:
            raise ValueError(f"stack count cannot be negative: {count}")

        if for_pick:
            return base_z + (count - 1) * chip_h
        else:
            return base_z + count * chip_h

    def _get_idle_pose(self) -> PoseObject:
        try:
            p = self.cfg["piecewalker"]["idle_pose"]
            return PoseObject(
                float(p["x"]), float(p["y"]), float(p["z"]),
                float(p["roll"]), float(p["pitch"]), float(p["yaw"])
            )
        except KeyError as e:
            raise ValueError(f"Missing config key for idle_pose in cfg file: {e}") from e

    def __init__(self) -> None:
        """
        Never call the constructor directly.
        Use the ned2_provider to instantiate it as a singleton.
        """
        print("Initializing Ned2 instance...")
        self.robot: NiryoRobot | None = None
        self.unplaced_chips_count = 9
        self.removed_chips_count = 0

        self.cfg = get_config()
        self.args = parse_args()

        raw = getattr(self.args, "calibration_file", None)
        self.calibration_file_path: Path | None = (
            raw if raw is None
            else Path(raw).expanduser().resolve()
        )
        self.calibration: Dict[str, Any] = parse_calibration_toml(self.calibration_file_path)

        self.pick_z_offset: float = float(self.cfg["piecewalker"].get("pick_z_offset", 0.0))
        self.place_z_offset: float = float(self.cfg["piecewalker"].get("place_z_offset", 0.0))

        # Connect robot
        ip = self.cfg["piecewalker"]["robot_ip"]
        print("Connecting to robot...")
        self.robot = NiryoRobot(ip)
        print("Calibrating and updating tool...")
        self.robot.calibrate_auto()
        self.robot.update_tool()

        self._idle_pose = self._get_idle_pose()
        print("Moving into idle pose...")
        self.robot.move_pose(self._idle_pose)

    def __del__(self) -> None:
        # Fail-safe, don't rely on it, Python doesn't guarantee that __del__ will 100% be called.
        self.close()
