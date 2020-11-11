from geometry import *
from obstacle import *
from robot import Robot 
from random import randint

# Squared world
class World:
	def __init__(self, width = 4.0, height = 4.0, robot = Robot()):
		self.corners = [Point(-height, -width), Point(-height, width), Point(height, width), Point(height, -width)]
		self.max_x = max([corner.x for corner in self.corners])
		self.max_y = max([corner.y for corner in self.corners])
		self.min_x = min([corner.x for corner in self.corners])
		self.min_y = min([corner.y for corner in self.corners])
		self.robot = robot
		self.obstacles = list()
		self.edges = list()
		self.__define_edges()
		self.__place_robot()
		self.obstacle_length = Obstacle.length

	def __repr__(self):
		repr_ = "--World--"
		repr_ += "\nEdges:\n"
		if self.edges:
			for idx, edge in enumerate(self.edges):	
				if idx == len(self.edges) - 1:
					repr_ += '['+str(idx)+'] ' + str(edge)
				else:
					repr_ += '['+str(idx)+'] ' + str(edge) + '\n'
		repr_ += "\nObstacles:\n"
		if self.obstacles:
			for idx, obstacle in enumerate(self.obstacles):	
				if idx == len(self.obstacles) - 1:
					repr_ += '['+str(idx)+'] ' + str(obstacle)
				else:
					repr_ += '['+str(idx)+'] ' + str(obstacle) + '\n'
		return repr_

	def __define_edges(self):
		for idx, point in enumerate(self.corners):
			if idx == len(self.corners) - 1:
				self.edges.append(Edge(self.corners[idx], self.corners[0]))
			else:
				self.edges.append(Edge(self.corners[idx], self.corners[idx + 1]))

	def __place_robot(self):
		self.robot.x, self.robot.y, self.robot.theta = -1.5, -0.5, 0.0

	def distance_to_nearest_wall(self, x, y):
		return min((abs(x - self.min_x), abs(x - self.max_x), abs(y - self.min_y), abs(y - self.max_y)))
		
	def set_obstacles_randomly(self, quantity = 4, type_ = 'round', dynamics = None):
		x, y = -10000, -10000	
		for _ in range(quantity):
			if type_ == 'round':
				while True:
					counter_obstacles = 0
					x = randint(self.min_x, self.max_x)
					y = randint(self.min_y, self.max_y)
					if self.inside_world(x,y) and self.distance_to_nearest_wall(x,y) > self.robot.diameter:
						for obstacle in self.obstacles:
							if x != obstacle.x and y != obstacle.y:
								counter_obstacles += 1
					if counter_obstacles == len(self.obstacles):
						break
				self.obstacles.append(RoundObstacle(x = x, y = y, dynamics = dynamics))
			else:
				while True:
					counter_obstacles = 0
					x = randint(self.min_x, self.max_x)
					y = randint(self.min_y, self.max_y)
					if self.inside_world(x,y) and self.distance_to_nearest_wall(x,y) > self.robot.diameter:
						for obstacle in self.obstacles:
							if x != obstacle.x and y != obstacle.y:
								counter_obstacles += 1
					if counter_obstacles == len(self.obstacles):
						break
				self.obstacles.append(SquaredObstacle(x = x, y = y, dynamics = dynamics))

	def add_obstacle(self, x, y, type_ = 'round', dynamics = None):
		if not self.inside_world(x,y):
			raise Exception('Invalid x, y. Not inside the boundaries of the map')
		distance_to_nearest_wall = self.distance_to_nearest_wall(x,y) - self.obstacle_length
		if distance_to_nearest_wall <= self.robot.diameter:
			raise Exception('Obstacle too close to the wall '+'('+str(distance_to_nearest_wall)+'m)'+'. Robot could get stuck between the wall and the obstacle')
		self.obstacles.append(RoundObstacle(x = x, y = y, dynamics = dynamics))

	def inside_world(self, x, y):
		return not (x > self.max_x or x < self.min_x or y > self.max_y or y < self.min_y)



