import numpy as np
from geometry import *
from obstacle import *
import math
import matplotlib.pyplot as plt

HIGHEST_NUMBER = 1e10

class Robot:
	radius = 0.09
	diameter = radius * 2.0
	def __init__(self, dt = 0.04):
		self.__x, self.__y, self.__theta = 0.0, 1.0, np.pi*0.5
		self.__position = Point(self.__x, self.__y)
		self.__dt = dt # control period
		self.obstacle_radius = RoundObstacle.diameter * 0.5
		self.__lidar = LIDAR(frequency = 50e3, origin = self.__position, obstacle_radius = self.obstacle_radius)

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
	@property
	def rotation(self):
		return self.__rotation
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
		min_distance = self.__lidar.min_distance
		max_distance = self.__lidar.max_distance
		to_rad = np.pi / 180.0
		
		for angle in range(-20,20):
			angle_rad = angle * to_rad
			# Laser line
			xi = self.__x + min_distance * np.cos(angle_rad + self.__theta)
			yi = self.__y + min_distance * np.sin(angle_rad + self.__theta)
			xf = self.__x + max_distance * np.cos(angle_rad + self.__theta)
			yf = self.__y + max_distance * np.sin(angle_rad + self.__theta)
			xi_back = self.__x + min_distance * np.cos(angle_rad + self.__theta + 180 * to_rad)
			yi_back = self.__y + min_distance * np.sin(angle_rad + self.__theta + 180 * to_rad)
			laser_line = Line(Point(x = xi, y = yi), Point(x = xf , y = yf))
			slope = laser_line.slope
			intercept = laser_line.intercept
			closest_obstacle = []
			minimum = HIGHEST_NUMBER

			# Compute distances to obstacles
			for obstacle in obstacles:
				xo = obstacle.x
				yo = obstacle.y
				distance_front = self.distance_between_points(xi, yi, xo, yo)
				distance_back = self.distance_between_points(xi_back, yi_back, xo, yo)
				distance = None
				if self.__lidar.in_sight(xi, yi, xo, yo) and distance_front < distance_back:
					if not slope:
						discriminant = self.obstacle_radius**2 - (intercept - xo)**2
						if discriminant > 0:
							x1 = x2 = intercept
							y1 = np.sqrt(discriminant) + yo
							y2 = -np.sqrt(discriminant) + yo
							distance = self.compute_distance(x1, y1, x2, y2, laser_x_front = xi, laser_y_front = yi, laser_x_back = xi_back, laser_y_back = yi_back, line = laser_line, circle = obstacle)
					else:
						discriminant = (self.obstacle_radius** 2) * (1 + slope ** 2) - (yo - slope * xo - intercept) ** 2
						if discriminant > 0:
							x1 = (xo + yo * slope - intercept * slope + np.sqrt(discriminant)) / (1 + slope ** 2)
							x2 = (xo + yo * slope - intercept * slope - np.sqrt(discriminant)) / (1 + slope ** 2)
							y1 = (intercept + xo * slope + yo * slope ** 2 + slope * np.sqrt(discriminant)) / (1 + slope ** 2)
							y2 = (intercept + xo * slope + yo * slope ** 2 - slope * np.sqrt(discriminant)) / (1 + slope ** 2)
							distance = self.compute_distance(x1, y1, x2, y2, laser_x_front = xi, laser_y_front = yi, laser_x_back = xi_back, laser_y_back = yi_back, line = laser_line, circle = obstacle)		
					if distance:
						if distance < minimum:
							minimum = distance
							closest_obstacle = obstacle

			# else, check if laser hit map corner
			#for edge in edges:
			
			self.__lidar.lasers[angle] = np.clip(minimum, self.__lidar.min_distance, self.__lidar.max_distance)
		return self.__lidar.lasers
	def compute_distance(self, x1, y1, x2, y2, laser_x_front, laser_y_front, laser_x_back, laser_y_back, line, circle):
		minimum_distance = HIGHEST_NUMBER
		right_points = None
		if circle.intersects(x1, y1) and line.intersects(x1, y1):
			distance = self.distance_between_points(x1, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x1, y1

		if circle.intersects(x1, y2) and line.intersects(x1, y2):
			distance = self.distance_between_points(x1, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x1, y2

		if circle.intersects(x2, y1) and line.intersects(x2, y1):
			distance = self.distance_between_points(x2, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x2, y1

		if circle.intersects(x2, y2) and line.intersects(x2, y2):
			distance = self.distance_between_points(x2, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x2, y2
		return minimum_distance

	@staticmethod
	def distance_between_points(x1, y1, x2, y2):
		return np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))

class LIDAR:
	n_lasers = 360
	min_distance = 0.12
	max_distance = 3.5
	def __init__(self, frequency, origin, obstacle_radius):
		self.frequency = frequency
		self.origin = origin
		self.obstacle_radius = obstacle_radius
		self.lasers = [0.0 for angle in range(self.n_lasers)]

	def in_sight(self, x_sight, y_sight, x_obstacle, y_obstacle):
		return (np.sqrt((x_obstacle - x_sight)**2 + (y_obstacle - y_sight)**2) - self.obstacle_radius) <= self.max_distance

	
		



