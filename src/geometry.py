class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __repr__(self):
		return 'x:'+str(self.x)+', y:'+str(self.y)
		
class Edge:
	def __init__(self, point1, point2):
		self.points = (point1, point2)
		try:
			self.slope = (point2.y -  point1.y) / (point2.x - point1.x)
			self.intercept = point2.y - self.slope * point2.x

		except ZeroDivisionError:
			self.slope = None
			self.intercept = point2.x
		
	def __repr__(self):
		return str(self.points)
		
class Line:
	pass
	