#!/usr/bin/env python3

import rospy
import numpy as np
import tf

from rospy.numpy_msg import numpy_msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

from world import World
from agent import Agent
import time

class GridWorldSimulation():

    def __init__(self, world):

        # Get map size
        self.world = world

        # ROS Subscribers
        # click_sub = rospy.Subscriber("clicked_point", PointStamped, update_reference)
        self.robot_sub = rospy.Subscriber("robot_state", Pose, self.update_robot)

        # ROS Publishers
        self.robot_marker_publisher = rospy.Publisher('robot_marker', Marker, queue_size=1)
        self.angle_marker_publisher = rospy.Publisher('angle_marker', PoseStamped, queue_size=1)

        # Setup tf tree and rviz objects
        self.world_tf = tf.TransformBroadcaster()
        self.robot_position_marker = Marker()
        self.robot_angle_marker = PoseStamped()

        # Initialize robot position marker
        self.robot_position_marker.header.frame_id = "robot_frame"
        self.robot_position_marker.type = self.robot_position_marker.CYLINDER
        self.robot_position_marker.action = self.robot_position_marker.ADD
        self.robot_position_marker.scale.x = 1.0
        self.robot_position_marker.scale.y = 1.0
        self.robot_position_marker.scale.z = 0.1
        self.robot_position_marker.color.a = 1.0
        self.robot_position_marker.color.r = 0.0
        self.robot_position_marker.color.g = 0.0
        self.robot_position_marker.color.b = 1.0
        self.robot_position_marker.pose.position.x = 0.0
        self.robot_position_marker.pose.position.y = 0.0
        self.robot_position_marker.pose.position.z = -0.1
        self.robot_position_marker.pose.orientation.x = 0.0
        self.robot_position_marker.pose.orientation.y = 0.0
        self.robot_position_marker.pose.orientation.z = 0.0
        self.robot_position_marker.pose.orientation.w = 1.0

        # Initialize robot orientation marker
        self.robot_angle_marker.header.frame_id = "robot_frame"
        self.robot_angle_marker.pose.position.x = 0.0
        self.robot_angle_marker.pose.position.y = 0.0
        self.robot_angle_marker.pose.position.z = 0.0
        self.robot_position_marker.pose.orientation.x = 0.0
        self.robot_position_marker.pose.orientation.y = 0.0
        self.robot_position_marker.pose.orientation.z = 0.0
        self.robot_position_marker.pose.orientation.w = 1.0

        # Initial map update
        self.update_robot()

    # This function receives a Pose ROS msg and updates the corresponding transformation
    def update_robot(self):

        robot_marker_position_x = self.world.robot.x
        robot_marker_position_y = self.world.robot.y
        robot_quat = quaternion_from_euler(0, 0, self.world.robot.theta)

        # Sends transformations and publishes Rviz graphical objects
        self.world_tf.sendTransform((robot_marker_position_x,robot_marker_position_y,0),robot_quat,rospy.Time.now(),"robot_frame", "world")
        self.robot_marker_publisher.publish(self.robot_position_marker)
        self.angle_marker_publisher.publish(self.robot_angle_marker)

if __name__ == '__main__':
    try:
        # Initialize world
        sim_dt = 0.01
        sim_freq = 1/sim_dt
        world = World(dt = sim_dt, width = 4, height = 4)

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

        # Initialize node
        rospy.init_node('graphics_broadcaster', anonymous=True)
        rospy.loginfo("Starting robot simulation...")
        
        # Initialize simulation object
        turtleSimulation = GridWorldSimulation(world)

        # Main ROS loop
        rate = rospy.Rate(sim_freq)  # 10hz
        while not rospy.is_shutdown():
            v = 0
            w = np.random.rand()
            agent.act(v, w)

            turtleSimulation.update_robot()
            print(world.robot)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
