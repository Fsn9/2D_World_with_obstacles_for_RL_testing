from world import World
from agent import Agent
from robot import Robot

# Initialize robot and world
world = World(dt = 1, width = 4, height = 4)

# Add obstacles
world.add_obstacle(1.0, 1.0)
world.add_obstacle(1.0, 0.0)
world.add_obstacle(1.0, -1.0)
world.add_obstacle(0.0, 1.0)
world.add_obstacle(0.0, 0.0)
world.add_obstacle(0.0, -1.0)
world.add_obstacle(-1.0, 1.0)
world.add_obstacle(-1.0, 0.0)
world.add_obstacle(-1.0, -1.0)

# Give world and robot to the agent
agent = Agent(world = world)

# Step robot
v = 0.1
w = 0.05
samples = 10

for t in range(samples):
	agent.act(v, w)