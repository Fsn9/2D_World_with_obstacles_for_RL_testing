class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __repr__(self):
		return '('+str(self.x)+')'+'('+str(self.y)+')'
		
class Edge:
	def __init__(self, point1, point2):
		self.points = (point1, point2)
	def __repr__(self):
		return 'Edge: '+ str(self.points)
		
class Line:
	pass
	