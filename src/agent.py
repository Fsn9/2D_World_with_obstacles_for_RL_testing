class Agent:
	def __init__(self, world, robot):
		self.__world = world
		self.__robot = robot

	def act(self, v, w):
		observation, reward, terminal, debug = self.__world.move_robot(v,w)
		#print(observation, reward, terminal, debug)
		#print(self.__robot)