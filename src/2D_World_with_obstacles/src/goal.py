from geometry import Circle

class Goal(Circle):
	diameter = 0.25
	def __init__(self, x = 0.0, y = 0.0, dynamics = None):
		super().__init__(x = x, y = y, radius = self.diameter * 0.5)
		self.__dynamics = dynamics
		self.__radius = self.diameter * 0.5

	@property
	def radius(self):
		return self.__radius