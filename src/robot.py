class Robot:
	radius = 0.09 # meters
	diameter = radius * 2.0
	def __init__(self):
		self.__x, self.__y, self.__theta = 0.0, 0.0, 0.0
	def __repr__(self):
		return '--Robot--' + '\nx: ' + str(self.__x) + '\ny: ' + str(self.__y) + '\ntheta: ' + str(self.__theta) + '\n'
	@property
	def x(self):
		return self.__x
	@x.setter
	def x(self, x):
		self.__x = x
	@property
	def y(self):
		return self.__y
	@y.setter
	def y(self, y):
		self.__y = y
	@property
	def theta(self):
		return self.__theta	
	@theta.setter
	def theta(self, theta):
		self.__theta = theta