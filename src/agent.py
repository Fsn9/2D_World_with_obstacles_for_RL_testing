#from dqn import DQN

class Agent:
	def __init__(self, world):
		self.__world = world

	def act(self, v, w):
		#dqn_learner = DQN(first_laser_scan = first_laser_scan, first_food_position = first_food_position, robot_pose = robot_initial_pose, action_publisher = action_publisher)
		observation, reward, terminal, debug = self.__world.move_robot(v,w)