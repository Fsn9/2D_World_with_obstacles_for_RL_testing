import numpy as np
from geometry import *
from obstacle import *
import math
import matplotlib.pyplot as plt
import time

HIGHEST_NUMBER = 1e10
TO_DEG = 57.29577
TO_RAD = 0.01745

class Robot(Circle):
	radius = 0.09
	diameter = radius * 2.0
	def __init__(self, dt = 0.04):
		super().__init__(x = 0, y = 1.5, radius = self.radius)
		#super().__init__(x = 0, y = 0, radius = self.radius)
		self.__theta = np.pi * 0.5 
		self.__position = Point(self._x, self._y)
		self.__dt = dt # control period
		self.obstacle_radius = RoundObstacle.diameter * 0.5
		self.__lidar = LIDAR(frequency = 5, origin = self.__position, obstacle_radius = self.obstacle_radius)

	def __repr__(self):
		return 'Robot pose' + ' | x: ' + str(self._x) + ' | y: ' + str(self._y) + ' | theta: ' + str(self.__theta)
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

	def set_pose(self, x, y, theta):
		self._x, self._y, self.__theta = x, y, theta

	def move(self, v, w):
		v_left = v + w * self.diameter * 0.5
		v_right = v - w * self.diameter * 0.5
		dd = (v_left + v_right) * 0.5
		dth = (v_left - v_right) / self.diameter
		self._x += dd * np.cos(self.__theta + dth * 0.5) * self.dt
		self._y += dd * np.sin(self.__theta + dth * 0.5) * self.dt
		self.__theta = self.normalize_angle(self.__theta)
		self.__theta += dth * self.dt
		return self._x, self._y, self.__theta

	def update_lidar(self, obstacles, edges):
		start = time.time()
		rot = Rotation(self.__theta * TO_DEG)

		## Walls
		# Angle ranges to map vertexes
		betas = np.empty((len(edges), 2))
		for idx, edge in enumerate(edges):
			points = rot(edge.points)
			xy = rot(np.array([self._x, self._y]))
			betas[idx][0], betas[idx][1] = np.arctan2(points[0][1] - xy[0][1], points[0][0] - xy[0][0]), np.arctan2(points[1][1] - xy[0][1], points[1][0] - xy[0][0])
		betas = np.rad2deg(betas).astype(int)
		betas = np.where(betas < 0, betas + 360, betas)

		# Angle differences from the lower distance to wall for each angle range
		differences = list()
		for idx, beta in enumerate(betas):
			if idx == len(betas) - 1:	
				differences += self.difference_vector(beta[0], beta[1])
			else:
				differences += self.difference_vector(beta[0], beta[1])[:-1]
		cos_differences = np.cos(np.deg2rad(np.array(differences)))

		# Compute distances to walls
		angle = 0
		for idx, edge in enumerate(edges):
			if edge.is_horizontal():
				dmin = abs(self._y - edge.intercept)
			else:
				dmin = abs(self._x - edge.intercept)

			if betas[idx][0] > betas[idx][1]:	
				angular_range = betas[idx][1] - betas[idx][0] + 360
				remain = self.__lidar.n_lasers - betas[idx][0]
				self.__lidar.lasers[betas[idx][0]:] = np.clip((dmin / cos_differences[angle:angle+remain] - self.radius).tolist(), self.__lidar.min_distance, self.__lidar.max_distance)
				self.__lidar.lasers[:betas[idx][1]] = np.clip((dmin / cos_differences[angle+remain:angle+remain+betas[idx][1]] - self.radius).tolist(), self.__lidar.min_distance, self.__lidar.max_distance)
				angle += angular_range
			else:
				angular_range = betas[idx][1] - betas[idx][0]
				self.__lidar.lasers[betas[idx][0]:betas[idx][1]] = np.clip((dmin / cos_differences[angle:angle+angular_range] - self.radius).tolist(), self.__lidar.min_distance, self.__lidar.max_distance)
				angle += angular_range

		## Obstacles
		# Get angle positions of the obstacles
		vectors_obstacles = np.empty((len(obstacles),2))
		for idx, obstacle in enumerate(obstacles):
			vectors_obstacles[idx] = rot((np.array([obstacle.x, obstacle.y]) - np.array([self._x, self._y])).reshape(1,2))
		distances = np.linalg.norm(vectors_obstacles, axis = 1)
		deltas = np.rad2deg(np.arctan2(vectors_obstacles[:,1], vectors_obstacles[:,0]))
		deltas = np.where(deltas < 0, deltas + 360, deltas).astype(int)
		alphas = np.rad2deg(np.arctan(self.obstacle_radius / distances)).astype(int)

		for i in range(len(obstacles)):
			# if faraway
			if (distances[i] - self.obstacle_radius - self.radius) > self.__lidar.max_distance:
				self.__lidar.lasers[deltas[i] - alphas[i]:deltas[i] + alphas[i]] = [self.__lidar.max_distance] * alphas[i] * 2
			# if near
			else:
				angles = [angle for angle in range(deltas[i] - alphas[i], deltas[i] + alphas[i] + 1)]
				for j, angle in enumerate(angles):
					distance = self.__lidar.max_distance
					xi, yi, xf, yf = self.__lidar.get_laser_points(angles[j], self._x, self._y, self.__theta)
					laser_line = Line(Point(xi, yi), Point(xf,yf))
					points = laser_line.intersects_circle(obstacles[i])
					if points: 
						distance = self.compute_distance(points[0], points[1], points[2], points[3], xi, yi, laser_line, obstacles[i])
						if self.__lidar.lasers[angle] != 0.0:
							self.__lidar.lasers[angle] = np.clip(min(distance, self.__lidar.lasers[angle]), self.__lidar.min_distance, self.__lidar.max_distance)
						else:
							self.__lidar.lasers[angle] = np.clip(distance, self.__lidar.min_distance, self.__lidar.max_distance)
		#self.plot_laser_distances(-180,179)
		end = time.time()
		#print('update lidar time:', (end-start) * 1000,'ms')
		return self.__lidar.lasers

	@staticmethod	
	def difference_vector(low, high):
		if high < low:
			high += 360
		avg = (high - low) * 0.5
		differences = list()
		rng = round(avg)
		for i in range(-rng, rng + 1):
			if i == 0 and avg%1 != 0:
				continue
			differences.append(i)
		return differences

	@staticmethod
	def normalize_angle(angle):
		if angle > math.pi:
			angle = angle - math.pi
		elif angle < - math.pi:
			angle = angle + math.pi
		return angle

	def compute_distance(self, x1, y1, x2, y2, laser_x_front, laser_y_front, line, circle):
		minimum_distance = HIGHEST_NUMBER
		if circle.intersects(x1, y1) and line.intersects(x1, y1):
			distance = self.__lidar.distance_between_points(x1, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
		if circle.intersects(x1, y2) and line.intersects(x1, y2):
			distance = self.__lidar.distance_between_points(x1, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
		if circle.intersects(x2, y1) and line.intersects(x2, y1):
			distance = self.__lidar.distance_between_points(x2, y1, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
		if circle.intersects(x2, y2) and line.intersects(x2, y2):
			distance = self.__lidar.distance_between_points(x2, y2, laser_x_front, laser_y_front)
			if distance < minimum_distance:
				minimum_distance = distance
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
	def __init__(self, frequency, origin, obstacle_radius):
		self.frequency = frequency
		self.origin = origin
		self.obstacle_radius = obstacle_radius
		self.lasers = [0.0 for angle in range(self.n_lasers)]

	def in_between(self, xi, xm, xf):
		return xi <= xm <= xf or xf <= xm <= xi

	def in_sight(self, x_sight, y_sight, x_forward, y_forward, x_object, y_object, obj):
		if isinstance(obj, Circle):
			return (self.in_between(x_sight, x_object, x_forward) and self.in_between(y_sight, y_object, y_forward)) and (self.distance_between_points(x_object, y_object, x_sight, y_sight) - obj.radius) <= self.max_distance
		elif isinstance(obj, Line):
			return (self.in_between(x_sight, x_object, x_forward) and self.in_between(y_sight, y_object, y_forward)) and self.distance_between_points(x_object, y_object, x_sight, y_sight) <= self.max_distance 

	def get_laser_points(self, angle, x, y, theta):
		angle_rad = angle * TO_RAD
		cos = np.cos(angle_rad + theta)
		sin = np.sin(angle_rad + theta)
		xi = x + self.min_distance * cos
		yi = y + self.min_distance * sin
		xf = x + self.max_distance * cos
		yf = y + self.max_distance * sin
		return xi, yi, xf, yf

	@staticmethod
	def distance_between_points(x1, y1, x2, y2):
		return np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))

	
		



