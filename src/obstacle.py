from geometry import *
class Obstacle(object):
	length = 0.25
	def __init__(self, x = 0.0, y = 0.0, dynamics = None):
		self.__position = Point(x,y)
		self._dynamics = dynamics

	def __repr__(self):
		return 'dynamics: ' + str(self._dynamics) + '\nposition: '+str(self.__position)

	@property
	def x(self):
		return self.__position.x

	@property
	def y(self):
		return self.__position.y

	@property
	def position(self):
		return self.__position

	def update_position(self, x, y):
		self._x = x
		self._y = y

class RoundObstacle(Obstacle):
	def __init__(self, x = 0.0, y = 0.0, dynamics = None):
		super().__init__(x, y, dynamics)

class SquaredObstacle(Obstacle):
	def __init__(self, length = 0.25, x = 0.0, y = 0.0, dynamics = None):
		super().__init__(x, y, dynamics)
