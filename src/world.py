from geometry import *
from obstacle import *
from robot import Robot 
from random import randint

# Squared world
class World(Square):
	def __init__(self, dt = 1, width = 4.0, height = 4.0):
		super().__init__(Point(-height * 0.5, -width * 0.5), Point(-height * 0.5, width * 0.5), Point(height * 0.5, width * 0.5), Point(height * 0.5, -width * 0.5))
		self.robot = Robot(dt = dt)
		self.obstacles = list()
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
				self.obstacles.append(RoundObstacle(x = x, y = y, dynamics = dynamics))
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
				self.obstacles.append(SquaredObstacle(x = x, y = y, dynamics = dynamics))

	def add_obstacle(self, x, y, type_ = 'round', dynamics = None):
		if not self.inside_world(x,y):
			raise Exception('Invalid x, y. Not inside the boundaries of the map')
		distance_to_nearest_wall = self.distance_to_nearest_wall(x,y) - self.obstacle_length
		if distance_to_nearest_wall <= self.robot.diameter:
			raise Exception('Obstacle too close to the wall '+'('+str(distance_to_nearest_wall)+'m)'+'. Robot could get stuck between the wall and the obstacle')
		self.obstacles.append(RoundObstacle(x = x, y = y, dynamics = dynamics))

	def collided_with_obstacle(self, x, y):
		self.obstacle_radius = self.obstacle_length * 0.5
		for obstacle in self.obstacles:
			if obstacle.inside(x,y):
				return True

	def inside_world(self, x, y):
		return not (x > self._max_x or x < self._min_x or y > self._max_y or y < self._min_y)

	def move_robot(self, v, w):
		x, y, theta = self.robot.move(v, w)
		lasers = self.robot.update_lidar(self.obstacles, self._edges)
		self.robot.plot_laser_distances(-179,180)
		print(self.robot)
		reward = 0.0
		terminal = False
		observation = []
		debug = []

		# Out of the world
		if not self.inside_world(x, y):
			terminal = True

		# Collision with obstacle
		if self.collided_with_obstacle(x, y):
			terminal = True

		return observation, reward, terminal, debug




