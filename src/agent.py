class Agent:
	def __init__(self, world):
		self.__world = world

	def act(self, v, w):
		observation, reward, terminal, debug = self.__world.move_robot(v,w)