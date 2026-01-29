from pyniryo import NiryoRobot, PoseObject

position_index = {}
robot: NiryoRobot
remove_pos: PoseObject


def init(curr_ip: str, position_indices: dict[int, PoseObject], remove_position: PoseObject) -> None:
    global position_index, robot, remove_pos
    position_index = position_indices
    remove_pos = remove_position
    robot = NiryoRobot(curr_ip)
    robot.calibrate_auto()
    robot.update_tool()


def play_move(start: int, stop: int):
    robot.pick_and_place(position_index[start], position_index[stop])


def remove_piece(field: int):
    robot.pick_and_place(position_index[field], remove_pos)
