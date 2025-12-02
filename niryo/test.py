from pyniryo import PoseObject

import robot_interface

current_ip = "10.132.157.172"
positional_indices = {

}
remove_pos =  PoseObject(0.2593, 0.1302, 0.085, -1.4615, 1.4586, -1.1148)

robot_interface.init(current_ip, positional_indices, remove_pos)
robot_interface.play_move(4, 5)