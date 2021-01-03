from dqn import DQN
from random import choice

class Agent:
	def __init__(self, world, brain_name):
		self.__world = world
		self.brain_name = brain_name
		if brain_name == 'Q-learning':
			pass
		elif brain_name == 'DQN':
			self.observations = list()
			self.next_observations = list()
			self.brain = DQN()
		else:
			raise Exception('Not valid algorithm. Only DQN or Q-learning')
		# Constant linear velocity
		self.linear_velocity = self.brain.hp['control']['linear_velocity']
		# Memory of the last action decided
		self.action = choice(self.brain.action_space)
		self.terminal = False, None

	def act(self):
		if self.brain.actual_episodes > 0:
			self.action = self.brain.run(self.__world.observe(), self.action, self.terminal)
			self.terminal = self.__world.move_robot(v = self.linear_velocity, w = self.action)
			print('Episodes:',self.brain.actual_episodes)
			#print('terminal',self.terminal)
			return True
		else:
			return False

	def __act_q_learning(self):
		pass