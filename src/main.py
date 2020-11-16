from world import World
from agent import Agent
from robot import Robot

# Initialize robot and world
robot = Robot(dt = 0.5)
world = World(robot = robot, width = 4, height = 4)

# Add obstacles
world.add_obstacle(2.0, 2.0)
world.add_obstacle(2.0, 0.0)
world.add_obstacle(2.0,-2.0)
world.add_obstacle(0.0, 2.0)
world.add_obstacle(0.0, 0.0)
world.add_obstacle(0.0,-2.0)
world.add_obstacle(-2.0, 2.0)
world.add_obstacle(-2.0, 0.0)
world.add_obstacle(-2.0,-2.0)


# Give world and robot to the agent
agent = Agent(world = world, robot = robot)

# Step robot
v = 0.15
w = 0.2
samples = 20
for t in range(samples):
	agent.act(v, w)