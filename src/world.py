from geometry import *
from robot import Robot 

class World:
	def __init__(self, robot = Robot()):
		self.corners = [Point(-3.0, -3.0), Point(-3.0, 3.0), Point(3.0, 3.0), Point(3.0, -3.0)]
		self.robot = robot
		self.edges = list()
		self.define_edges()
		self.place_robot()
	
	def define_edges(self):
		for idx, point in enumerate(self.corners):
			if idx == len(self.corners) - 1:
				self.edges.append(Edge(self.corners[idx], self.corners[0]))
			else:
				self.edges.append(Edge(self.corners[idx], self.corners[idx + 1]))
				
	def __repr__(self):
		repr_ = "--World--\nedges:\n"
		for idx, edge in enumerate(self.edges):	
			if idx == len(self.edges) - 1:
				repr_ += '['+str(idx)+'] ' + str(edge)
			else:
				repr_ += '['+str(idx)+'] ' + str(edge) + '\n'
		return repr_

	def place_robot(self):
		self.robot.x, self.robot.y, self.robot.theta = -1.5, -0.5, 0.0