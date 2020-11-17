from geometry import *
from obstacle import *
from robot import Robot 
from random import randint

# Squared world
class World(Square):
	def __init__(self, dt = 1, width = 4.0, height = 4.0):
		super().__init__(Point(-width * 0.5, -height * 0.5 ), Point(width * 0.5, -height * 0.5), Point(width * 0.5, height * 0.5), Point(-width * 0.5, height * 0.5))
		# Edges
		self.objects = list()
		self.objects.extend(self._edges)
		# Robot
		self.robot = Robot(dt = dt)
		# Obstacles
		self.obstacle_length = RoundObstacle.diameter

	def __repr__(self):
		repr_ = "--World--"
		repr_ += "\nEdges:\n"
		if self._edges:
			for idx, edge in enumerate(self._edges):	
				if idx == len(self._edges) - 1:
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

	def distance_to_nearest_wall(self, x, y):
		return min((abs(x - self._min_x), abs(x - self._max_x), abs(y - self._min_y), abs(y - self._max_y)))
		
	def set_obstacles_randomly(self, quantity = 4, type_ = 'round', dynamics = None):
		x, y = -10000, -10000	
		for _ in range(quantity):
			if type_ == 'round':
				while True:
					counter_obstacles = 0
					x = randint(self._min_x, self._max_x)
					y = randint(self._min_y, self._max_y)
					if self.inside_world(x,y) and self.distance_to_nearest_wall(x,y) > self.robot.diameter:
						for obstacle in self.obstacles:
							if x != obstacle.x and y != obstacle.y:
								counter_obstacles += 1
					if counter_obstacles == len(self.obstacles):
						break
				self.objects.append(RoundObstacle(x = x, y = y, dynamics = dynamics))
			else:
				while True:
					counter_obstacles = 0
					x = randint(self._min_x, self._max_x)
					y = randint(self._min_y, self._max_y)
					if self.inside_world(x,y) and self.distance_to_nearest_wall(x,y) > self.robot.diameter:
						for obstacle in self.obstacles:
							if x != obstacle.x and y != obstacle.y:
								counter_obstacles += 1
					if counter_obstacles == len(self.obstacles):
						break
				self.objects.append(SquaredObstacle(x = x, y = y, dynamics = dynamics))

	def add_obstacle(self, x, y, type_ = 'round', dynamics = None):
		if not self.inside_world(x,y):
			raise Exception('Invalid x, y. Not inside the boundaries of the map')
		distance_to_nearest_wall = self.distance_to_nearest_wall(x,y) - self.obstacle_length
		if distance_to_nearest_wall <= self.robot.diameter:
			raise Exception('Obstacle too close to the wall '+'('+str(distance_to_nearest_wall)+'m)'+'. Robot could get stuck between the wall and the obstacle')
		self.objects.append(RoundObstacle(x = x, y = y, dynamics = dynamics))

	def collided(self, x, y):
		for obj in self.objects:
			if isinstance(obj, Circle):
				if self.robot.intersects_circle(obj):
					return True
			elif isinstance(obj, Line):
				if self.robot.intersects_line(obj):
					return True
			else:
				pass
			return False

	def inside_world(self, x, y):
		return not (x > self._max_x or x < self._min_x or y > self._max_y or y < self._min_y)

	def move_robot(self, v, w):
		x, y, theta = self.robot.move(v, w)
		lasers = self.robot.update_lidar(self.objects)
		self.robot.plot_laser_distances(-179,180)
		print(self.robot)
		reward = 0.0
		terminal = False
		observation = []
		debug = []

		# Collision with obstacle
		if self.collided(x,y) or not self.inside_world(x,y):
			terminal = True

		return observation, reward, terminal, debug




