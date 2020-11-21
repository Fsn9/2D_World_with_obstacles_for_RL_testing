from world import World
from agent import Agent
import time

# Initialize world
world = World(dt = 0.1, width = 4, height = 4)

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

# Give world to the agent
agent = Agent(world = world)

# Step robot
v = 0.1
w = 0
samples = 10
for t in range(samples):
	start = time.time()
	agent.act(v, w)
	end = time.time()
	print(end - start)