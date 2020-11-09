from geometry import *

class World:
	def __init__(self, robot):
		self.corners = [Point(-3.0, -3.0), Point(-3.0, 3.0), Point(3.0, 3.0), Point(3.0, -3.0)]
		self.robot = robot
		self.edges = list()
		self.define_edges()

		
	def define_edges(self):
		for idx, point in enumerate(self.corners):
			if idx == len(self.corners) - 1:
				self.edges.append(Edge(self.corners[idx], self.corners[0]))
			else:
				self.edges.append(Edge(self.corners[idx], self.corners[idx + 1]))
				
	def __repr__(self):
		repr = "World edges:\n"
		for idx, edge in enumerate(self.edges):	
			if idx == len(self.edges) - 1:
				repr += '['+str(idx)+'] ' + str(edge)
			else:
				repr += '['+str(idx)+'] ' + str(edge) + '\n'
		return repr