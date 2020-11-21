import numpy as np
from math import isclose
import time

class Rotation:
	def __init__(self, rotation_deg):
		self.__rotation = rotation_deg * np.pi / 180
		self.__rot_matrix = np.array([[np.cos(self.__rotation), -np.sin(self.__rotation)],[np.sin(self.__rotation), np.cos(self.__rotation)]])
	def __call__(self, x, y):
		res = np.dot(self.__rot_matrix, np.array([x,y]).reshape(2,1))
		return res[0][0], res[1][0]
	@property
	def rot_matrix(self):
		return self.__rot_matrix
	@property
	def rotation(self):
		return self.__rotation

class Point:
	def __init__(self, x, y):
		self.__x = x
		self.__y = y
	def __repr__(self):
		return 'x:'+str(self.x)+', y:'+str(self.y)
	def __add__(self, other):
		return self.__x + other.x, self.__y + other.y
	def __sub__(self, other):
		return self.__x - other,x, self.__y - other.y
	@property
	def x(self):
		return self.__x
	@property
	def y(self):
		return self.__y
	@x.setter
	def x(self, x):
		self.__x = x
	@y.setter
	def y(self, y):
		self.__y = y
	def rotate(self, rotation_deg):
		self.__x, self.__y = Rotation(rotation_deg)(self.__x, self.__y)
		
class Square:
	def __init__(self, point1, point2, point3, point4):
		self._corners = [point1, point2, point3, point4]
		self._max_x = max([corner.x for corner in self._corners])
		self._max_y = max([corner.y for corner in self._corners])
		self._min_x = min([corner.x for corner in self._corners])
		self._min_y = min([corner.y for corner in self._corners])
		self._edges = list()
		self.__define_edges()

	def __define_edges(self):
		for idx, point in enumerate(self._corners):
			if idx == len(self._corners) - 1:
				self._edges.append(Edge(self._corners[idx], self._corners[0]))
			else:
				self._edges.append(Edge(self._corners[idx], self._corners[idx + 1]))
class Line:
	highest_slope = 1e4
	def __init__(self, point1, point2):
		self._points = (point1, point2)
		denominator = point2.x - point1.x 
		numerator = point2.y - point1.y
		if not (denominator == 0 or isclose(point2.x, point1.x)):
			self._slope = (point2.y -  point1.y) / (point2.x - point1.x)
			self._intercept = point2.y - self._slope * point2.x
			if abs(self._slope) > self.highest_slope:
				self._slope = None
				self._intercept = point2.x
		else:
			self._slope = None
			self._intercept = point2.x	

	def __repr__(self):
		return str(self._points)+', slope: '+str(self._slope)+', intercept: '+str(self._intercept)

	def intersects(self, x, y):
		if not self._slope:
			return x == self._intercept or isclose(x, self._intercept, abs_tol = 1e-4)
		return y == self._slope * x + self._intercept or isclose(y, self._slope * x + self._intercept, abs_tol = 1e-4)

	def intersects_line(self, other):
		if self._slope is not None and other.slope is not None:
			x = (other.intercept - self._intercept) / (self._slope - other.slope)
			y = (other.intercept * self._slope - self._intercept * other.slope) / (self._slope - other.slope)
			return [x,y]
		elif self._slope is not None and not other.slope:
			return [other.intercept, self._slope * other.intercept + self._intercept]
		elif not self._slope and other.slope is not None:
			return [self._intercept, other.slope * self._intercept + other.intercept]
		else:
			return None

	def intersects_circle(self, circle):
		start = time.time()
		radius, xc, yc = circle.radius, circle.x, circle.y
		if not self._slope:
			discriminant = radius**2 - (self._intercept - xc)**2
			if discriminant <= 0:
				return False
			x1 = x2 = self._intercept
			y1 = np.sqrt(discriminant) + yc
			y2 = -np.sqrt(discriminant) + yc					
		else:
			discriminant = (radius ** 2) * (1 + self._slope ** 2) - (yc - self._slope * xc - self._intercept) ** 2
			if discriminant <= 0:
				return False
			x1 = (xc + yc * self._slope - self._intercept * self._slope + np.sqrt(discriminant)) / (1 + self._slope ** 2)
			x2 = (xc + yc * self._slope - self._intercept * self._slope - np.sqrt(discriminant)) / (1 + self._slope ** 2)
			y1 = (self._intercept + xc * self._slope + yc * self._slope ** 2 + self._slope * np.sqrt(discriminant)) / (1 + self._slope ** 2)
			y2 = (self._intercept + xc * self._slope + yc * self._slope ** 2 - self._slope * np.sqrt(discriminant)) / (1 + self._slope ** 2)
		end = time.time()
		return [x1,y1,x2,y2]

	@property
	def slope(self):
		return self._slope
	@property
	def intercept(self):
		return self._intercept
	@property
	def points(self):
		return self._points
	
class Edge(Line):
	def __init__(self, point1, point2):
		super().__init__(point1, point2)

class Circle:
	def __init__(self, x, y, radius):
		self._x, self._y, self._radius = x, y, radius
	def __repr__(self):
		return 'xc:'+str(self._x)+',yc:'+str(self._y)+',radius:'+str(self._radius)
	@property
	def x(self):
		return self._x
	@property
	def y(self):
		return self._y
	@property
	def radius(self):
		return self._radius
	def intersects(self, x, y):
		return (x - self._x)**2 + (y - self._y)**2 == self._radius**2 or isclose((x - self._x)**2 + (y - self._y)**2, self._radius**2, abs_tol = 1e-4)
	def inside(self, x, y):
		return (x - self._x)**2 + (y - self._y)**2 < self._radius**2 or self.intersects(x,y)
	def outside(self,x, y):
		return not self.inside(x,y)
	def intersects_circle(self, other):
		d = np.sqrt((other.x - self._x)**2 + (other.y - self._y)**2)
		return (self._radius + other.radius) > d and d > abs(self._radius - other.radius)
	def intersects_line(self, line):
		radius, xc, yc = self._radius, self._x, self._y
		start = time.time()
		if not line.slope:
			discriminant = radius**2 - (line.intercept - xc)**2
			if discriminant <= 0:
				end = time.time()
				print('diff:', end-start)
				return False
			x1 = x2 = line.intercept
			y1 = np.sqrt(discriminant) + yc
			y2 = -np.sqrt(discriminant) + yc					
		else:
			discriminant = (radius ** 2) * (1 + line.slope ** 2) - (yc - line.slope * xc - line.intercept) ** 2
			if discriminant <= 0:
				end = time.time()
				print('diff:', end-start)
				return False
			x1 = (xc + yc * line.slope - line.intercept * line.slope + np.sqrt(discriminant)) / (1 + line.slope ** 2)
			x2 = (xc + yc * line.slope - line.intercept * line.slope - np.sqrt(discriminant)) / (1 + line.slope ** 2)
			y1 = (line.intercept + xc * line.slope + yc * line.slope ** 2 + line.slope * np.sqrt(discriminant)) / (1 + line.slope ** 2)
			y2 = (line.intercept + xc * line.slope + yc * line.slope ** 2 - line.slope * np.sqrt(discriminant)) / (1 + line.slope ** 2)
		end = time.time()
		print('diff:', end-start)
		return [x1,y1,x2,y2]

