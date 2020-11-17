import numpy as np
from geometry import *
from obstacle import *
import math
import matplotlib.pyplot as plt

HIGHEST_NUMBER = 1e10

class Robot(Circle):
	radius = 0.09
	diameter = radius * 2.0
	def __init__(self, dt = 0.04):
		super().__init__(x = 0.0, y = 1.5, radius = self.radius)
		self.__theta = np.pi * 0.5 
		self.__position = Point(self._x, self._y)
		self.__dt = dt # control period
		self.obstacle_radius = RoundObstacle.diameter * 0.5
		self.__lidar = LIDAR(frequency = 50e3, origin = self.__position, obstacle_radius = self.obstacle_radius)

	def __repr__(self):
		return '--Robot--' + '\nx: ' + str(self._x) + ', y: ' + str(self._y) + ',theta: ' + str(self.__theta)
	@property
	def x(self):
		return self._x
	@property
	def y(self):
		return self._y
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
		self._x, self._y, self.__theta = x, y, theta

	def move(self, v, w):
		v_left = v + w * self.diameter * 0.5
		v_right = v - w * self.diameter * 0.5
		dd = (v_left + v_right) * 0.5
		dth = (v_left - v_right) / self.diameter
		self._x += dd * np.cos(self.__theta + dth * 0.5) * self.dt
		self._y += dd * np.sin(self.__theta + dth * 0.5) * self.dt
		self.__theta += dth * self.dt
		return self._x, self._y, self.__theta

	def update_lidar(self, map_objects):
		for angle in range(self.__lidar.n_lasers):
			minimum = HIGHEST_NUMBER
			xi, yi, xf, yf, xi_back, yi_back = self.__lidar.get_laser_points(angle, self._x, self._y, self.__theta)
			laser_line = Line(Point(x = xi, y = yi), Point(x = xf , y = yf))
			for obj in map_objects:
				distance = None
				if isinstance(obj, Circle):
					xo, yo = obj.x, obj.y
					if self.__lidar.in_sight(xi, yi, xi_back, yi_back, xo, yo, obj):
						points = laser_line.intersects_circle(obj)
						if points:
							x1, y1, x2, y2 = points
							distance = self.compute_distance(x1, y1, x2, y2, xi, yi, xi_back, yi_back, laser_line, obj)
				elif isinstance(obj, Line):
					points = laser_line.intersects_line(obj)
					if points:
						xe, ye = points
						if self.__lidar.in_sight(xi, yi, xi_back, yi_back, xe, ye, obj):
							distance = self.__lidar.distance_between_points(xi, yi, xe, ye)
				else:
					pass
				if distance:
					if distance < minimum:
						minimum = distance
			self.__lidar.lasers[angle] = np.clip(minimum, self.__lidar.min_distance, self.__lidar.max_distance)
		return self.__lidar.lasers

	def compute_distance(self, x1, y1, x2, y2, laser_x_front, laser_y_front, laser_x_back, laser_y_back, line, circle):
		minimum_distance = HIGHEST_NUMBER
		right_points = None
		if circle.intersects(x1, y1) and line.intersects(x1, y1):
			distance = self.__lidar.distance_between_points(x1, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x1, y1

		if circle.intersects(x1, y2) and line.intersects(x1, y2):
			distance = self.__lidar.distance_between_points(x1, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x1, y2

		if circle.intersects(x2, y1) and line.intersects(x2, y1):
			distance = self.__lidar.distance_between_points(x2, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x2, y1

		if circle.intersects(x2, y2) and line.intersects(x2, y2):
			distance = self.__lidar.distance_between_points(x2, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
				right_points = x2, y2
		return minimum_distance

	def plot_laser_distances(self, first_angle, last_angle):
		if 0 < abs(last_angle - first_angle) < self.__lidar.n_lasers and last_angle > first_angle:
			if first_angle < 0 and last_angle > 0:
				plt.plot([x for x in range(first_angle, last_angle)], self.__lidar.lasers[first_angle:] + self.__lidar.lasers[0:last_angle])
			else:
				plt.plot([x for x in range(first_angle, last_angle)], self.__lidar.lasers[first_angle:last_angle])
			plt.show()
		else:
			raise Exception('Angle range must be between 0 and 360 and first angle needs to be lower than last_angle')
class LIDAR:
	n_lasers = 360
	min_distance = 0.12
	max_distance = 3.5
	to_rad = np.pi / 180.0
	def __init__(self, frequency, origin, obstacle_radius):
		self.frequency = frequency
		self.origin = origin
		self.obstacle_radius = obstacle_radius
		self.lasers = [0.0 for angle in range(self.n_lasers)]

	def in_sight(self, x_sight, y_sight, x_sight_back, y_sight_back, x_object, y_object, object):
		distance_front = self.distance_between_points(x_sight, y_sight, x_object, y_object)
		distance_back = self.distance_between_points(x_sight_back, y_sight_back, x_object, y_object)
		if isinstance(object, Circle):
			return (np.sqrt((x_object - x_sight)**2 + (y_object - y_sight)**2) - object.radius) <= self.max_distance and distance_front < distance_back
		elif isinstance(object, Line):
			return (np.sqrt((x_object - x_sight)**2 + (y_object - y_sight)**2)) <= self.max_distance and distance_front < distance_back

	def get_laser_points(self, angle, x, y, theta):
		angle_rad = angle * self.to_rad
		xi = x + self.min_distance * np.cos(angle_rad + theta)
		yi = y + self.min_distance * np.sin(angle_rad + theta)
		xf = x + self.max_distance * np.cos(angle_rad + theta)
		yf = y + self.max_distance * np.sin(angle_rad + theta)
		xi_back = x + self.min_distance * np.cos(angle_rad + theta + 180 * self.to_rad)
		yi_back = y + self.min_distance * np.sin(angle_rad + theta + 180 * self.to_rad)
		return xi, yi, xf, yf, xi_back, yi_back

	@staticmethod
	def distance_between_points(x1, y1, x2, y2):
		return np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))

	
		



