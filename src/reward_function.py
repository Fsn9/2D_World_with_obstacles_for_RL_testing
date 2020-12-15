#!/usr/bin/env python

import rospy
from numpy import digitize, asscalar
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float64
from nav_msgs.msg import OccupancyGrid
from qlearning_training_simulation.msg import FeatureStateVector
from math import pi
from utils import Ramp, Exponential, StateSpace

class Rewarder:
    def __init__(self):
        self.reward = Float64()

        # Robot info
        self.robot_radius = rospy.get_param('/robot_radius')

        # Parameters
        self.min_distance_food = rospy.get_param('/min_distance_food')
        self.max_distance_food = rospy.get_param('/max_distance_map')
        self.distance_food_resolution = rospy.get_param('/distance_obstacle_resolution')
        self.max_angle_food = rospy.get_param('/max_angle_food')
        self.min_angle_food = rospy.get_param('/min_angle_food')
        self.angle_food_resolution = rospy.get_param('/angle_food_resolution')
        self.min_distance_obstacle = rospy.get_param('/min_distance_seen')
        self.max_distance_seen = rospy.get_param('/max_distance_seen') 
        self.min_distance_obstacle = rospy.get_param('/min_distance_seen') # Minimum distance to be seen by the robot
        self.alpha = rospy.get_param('/alpha') # percentage of maximum distance seen by the robot
        self.max_distance_obstacle = self.alpha * self.max_distance_seen # actual max distance seen by the robot
        self.distance_obstacle_resolution = rospy.get_param('/distance_obstacle_resolution') # resolution of distance to obstacle
        self.angle_window_deg = rospy.get_param('/angle_window') # angle_window seen by the robot
        self.angle_window = self.angle_window_deg * pi / 180
        self.angle_obstacle_resolution = rospy.get_param('/angle_obstacle_resolution') # resolution of angle to obstacle
        self.number_of_frames = rospy.get_param('/number_of_frames')
        self.recent_frame_index = self.number_of_frames - 1
        
        '''FOOD'''
        # distance  
        self.critical_distance_food = rospy.get_param('/critical_distance_food')
        self.food_big_prize = rospy.get_param('/food_prize')
        self.max_reward_distance_food = rospy.get_param('/max_reward_distance_food')
        self.min_reward_distance_food = rospy.get_param('/min_reward_distance_food')
        self.reward_distance_food = Ramp(y2 = self.max_reward_distance_food, y1 = self.min_reward_distance_food, x2 = self.min_distance_food, x1 =  self.max_distance_food)

        # angle
        self.critical_angle_food = rospy.get_param('/critical_angle_food') # degrees
        self.max_reward_angle_food = rospy.get_param('/max_reward_angle_food')
        self.min_reward_angle_food = rospy.get_param('/min_reward_angle_food')
        self.beta = rospy.get_param('/beta') # reward value factor from where the steering angle to the food starts to become acceptable
        self.acceptable_reward_value = self.min_reward_angle_food / self.beta
        self.reward_angle_food_exp1 = Exponential(y2 = self.min_reward_angle_food, y1 = self.acceptable_reward_value, x2 = -180 , x1 = -self.critical_angle_food)
        self.reward_angle_food1 = Ramp(y2 = self.acceptable_reward_value, y1 = self.max_reward_angle_food, x2 = -self.critical_angle_food, x1 = 0.0)
        self.reward_angle_food2 = Ramp(y2 = self.max_reward_angle_food, y1 = self.acceptable_reward_value, x2 = 0.0, x1 = self.critical_angle_food)
        self.reward_angle_food_exp2 = Exponential(y2 = self.acceptable_reward_value, y1 = self.min_reward_angle_food, x2 = self.critical_angle_food, x1 = 180)

        '''OBSTACLE'''
        # distance
        self.critical_distance_obstacle = rospy.get_param('/critical_distance_obstacle')
        self.max_reward_distance_obstacle = rospy.get_param('/max_reward_distance_obstacle')
        self.min_reward_distance_obstacle = rospy.get_param('/min_reward_distance_obstacle')
        self.collision_penalty = rospy.get_param('/collision_penalty')
        self.reward_distance_obstacle = Ramp(y2 = self.min_reward_distance_obstacle, y1 = self.max_reward_distance_obstacle, x2 = self.min_distance_obstacle, x1 = self.critical_distance_obstacle)

        # angle
        self.critical_angle_obstacle = rospy.get_param('/critical_angle_obstacle')
        self.max_reward_angle_obstacle = rospy.get_param('/max_reward_angle_obstacle')
        self.min_reward_angle_obstacle = rospy.get_param('/min_reward_angle_obstacle')
        self.reward_angle_obstacle1 = Ramp(y2 = self.max_reward_angle_obstacle, y1 = self.min_reward_angle_obstacle, x2 = -self.critical_angle_obstacle, x1 = 0.0)
        self.reward_angle_obstacle2 = Ramp(y2 = self.min_reward_angle_obstacle, y1 = self.max_reward_angle_obstacle, x2 = 0.0, x1 = self.critical_angle_obstacle)

    def compute_reward(self, laser_frames, polar_food_frames, terminal, ate_food):
        # terminal state
        if terminal:
            return self.collision_penalty
        # food
        if ate_food:
            return self.food_big_prize
        # distance food component
        distance_food = polar_food_frames[self.recent_frame_index][0]
        return self.reward_distance_food(distance_food) if distance_food > self.min_distance_food else self.food_big_prize