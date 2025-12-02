from pyniryo import NiryoRobot, JointsPosition, PoseObject

curr_ip: str = "10.132.157.172"
pickup_pos = PoseObject(0.2593, 0.1302, 0.085, -1.4615, 1.4586, -1.1148)
idle_pos = PoseObject(0.2098, 0.0967, 0.3431, -1.8319, 1.4669, -1.5402)

robot = NiryoRobot(curr_ip)
robot.calibrate_auto()
robot.update_tool()

robot.release_with_tool()
robot.move(pickup_pos)
robot.grasp_with_tool()

robot.move(idle_pos)
robot.move(pickup_pos)

robot.release_with_tool()

robot.move(idle_pos)

robot.close_connection()
