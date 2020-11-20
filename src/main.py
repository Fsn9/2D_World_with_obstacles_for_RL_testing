from world import World
from agent import Agent

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
samples = 1
for t in range(samples):
	agent.act(v, w)