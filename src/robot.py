import numpy as np
from geometry import *
from obstacle import Obstacle

class Robot:
	radius = 0.09
	diameter = radius * 2.0
	def __init__(self, dt = 0.04):
		self.__x, self.__y, self.__theta = -1.5, 0.0, 0.0
		self.__position = Point(self.__x, self.__y)
		self.__dt = dt # control period
		self.__lidar = LIDAR(frequency = 50e3, origin = self.__position)

	def __repr__(self):
		return '--Robot--' + '\nx: ' + str(self.__x) + '\ny: ' + str(self.__y) + '\ntheta: ' + str(self.__theta) + '\n'
	@property
	def x(self):
		return self.__x
	@property
	def y(self):
		return self.__y
	@property
	def theta(self):
		return self.__theta
	@property 
	def position(self):
		return self.__position
	@property
	def dt(self):
		return self.__dt
	@property
	def lidar(self):
		return self.__lidar
	def set_pose(self, x, y, theta):
		self.__x, self.__y, self.__theta = x, y, theta

	def move(self, v, w):
		v_left = v + w * self.diameter * 0.5
		v_right = v - w * self.diameter * 0.5
		dd = (v_left + v_right) * 0.5
		dth = (v_left - v_right) / self.diameter
		self.__x += dd * np.cos(self.__theta + dth * 0.5) * self.dt
		self.__y += dd * np.sin(self.__theta + dth * 0.5) * self.dt
		self.__theta += dth * self.dt
		return self.__x, self.__y, self.__theta

	def update_lidar(self, obstacles, edges):
		obstacles_length = Obstacle.length
		min_distance = self.__lidar.min_distance
		max_distance = self.__lidar.max_distance
		to_rad = np.pi / 180.0
		for angle in range(self.__lidar.n_lasers):
			angle_rad = angle * to_rad
			# Laser line
			xi = self.__x + min_distance * np.cos(angle_rad)
			yi = self.__y + min_distance * np.sin(angle_rad)
			xf = self.__x + max_distance * np.cos(angle_rad)
			yf = self.__y + max_distance * np.sin(angle_rad)
			line = Line(Point(x = xi, y = yi), Point(x = xf , y = yf))
			slope = line.slope
			intercept = line.intercept

			# Check collision with obstacles
			for obstacle in obstacles:
				xo = obstacle.x
				yo = obstacle.y
				discriminant = ((obstacles_length * 0.5) ** 2) * (1 + slope ** 2) - (yo - slope * xo - intercept) ** 2

				# if discriminant is positive then laser intercepted obstacle
				if discriminant > 0:
					## compute distance to obstacle
					pass
				# else, check if laser hit map corner
				else:
					## compute distance to map corner
					pass

class LIDAR:
	n_lasers = 360
	min_distance = 0.12
	max_distance = 3.5
	
	def __init__(self, frequency, origin):
		self.frequency = frequency
		self.origin = origin
		self.lasers = [0.0 for angle in range(self.n_lasers)]
