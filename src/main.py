from world import World

# xy referencial: x (up), y (left)

# Define world
world = World(width = 4, height = 4)

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