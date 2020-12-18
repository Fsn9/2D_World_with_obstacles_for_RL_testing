from geometry import *
 
class RoundObstacle(Circle):
	diameter = 0.25
	def __init__(self, x = 0.0, y = 0.0, dynamics = None):
		super().__init__(x = x, y = y, radius = self.diameter * 0.5)
		self.__dynamics = dynamics
		self.__radius = self.diameter * 0.5

	@property
	def radius(self):
		return self.__radius

class SquaredObstacle(Circle): #change to Square
	length = 0.25
	def __init__(self, x = 0.0, y = 0.0, dynamics = None):
		super().__init__(x, y, self.radius)
		self.__dynamics = dynamics
