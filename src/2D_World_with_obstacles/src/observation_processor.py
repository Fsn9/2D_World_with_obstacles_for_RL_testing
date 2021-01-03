#!/usr/bin/env python

# main libraries
import rospy
import numpy as np
from math import sqrt, sin, cos
# messages
#from geometry_msgs.msg import Pose2D
#from std_msgs.msg import Float64, Float32, Int32MultiArray
#from sensor_msgs.msg import LaserScan
#from dqn_training_simulation.msg import *
# auxiliar libraries
from utils import Ramp, CenterOfGravity

TO_DEG = 57.29577
TO_RAD = 0.01745

class ObservationProcessor:
    def __init__(self, hyperparameters):
        self.hp = hyperparameters
        # Initialize robot data
        self.robot_theta = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_radius = self.hp['robot']['radius']
        self.robot_diameter = self.robot_radius * 2.0

        # Initialize food data
        self.food_x = 0.0
        self.food_y = 0.0

        ''' States parameters '''
        # Lasers
        self.max_distance_seen = self.hp['control']['max_distance_seen']
        self.min_distance_seen = self.hp['control']['min_distance_seen'] # Minimum distance to be seen by the robot
        self.alpha = self.hp['state']['alpha'] # percentage of maximum distance seen by the robot
        self.max_distance_obstacle = self.alpha * self.max_distance_seen # actual max distance seen by the robot
        self.angle_window_deg = self.hp['control']['angle_window'] # angle_window seen by the robot
        self.angle_window = np.deg2rad(self.angle_window_deg)
        self.dimension_lasers = self.angle_window_deg * 2 
        
        # Food     
        self.dimension_food_polar = 2

        # Save complexity
        self.dimension = self.dimension_lasers // 2 + self.dimension_food_polar

        # Sensitivity of decimal places
        self.sensitivity = self.hp['math']['sensitivity']
        
    def process(self, lasers, food_x, food_y, robot_x, robot_y, robot_theta):
        # Save data
        self.lasers = lasers
        self.food_x, self.food_y, self.robot_x, self.robot_y, self.robot_theta = food_x, food_y, robot_x, robot_y, robot_theta

        # laser feature
        lasers = self.process_lasers()

        # food feature
        distance_food, angle_food = self.find_food()

        return tuple(lasers), (distance_food, angle_food)

    def process_goal_data(self, robot_x, robot_y, robot_theta, goal_x, goal_y):
        distance_goal = round(np.linalg.norm(np.array([robot_x, robot_y]) - np.array([goal_x, goal_y])), self.sensitivity)
        vector_food_in_frame_robot = self.rotate_vector_2D(np.array([goal_x - robot_x, goal_y - robot_y]), - robot_theta * 180 / np.pi)
        angle_goal = self.normalize_angle(round(np.arctan2(vector_food_in_frame_robot[1], vector_food_in_frame_robot[0]), self.sensitivity)) 
        return distance_goal, angle_goal

    def process_laser_data(self, lasers):
        return lasers[-self.angle_window_deg // 2:] + lasers[:self.angle_window_deg // 2]

    def process_lasers(self):
        positive_window, negative_window = self.lasers[0: self.angle_window_deg // 2], self.lasers[-self.angle_window_deg // 2: -1] + [self.lasers[len(self.lasers)- 1]]
        lasers = negative_window + positive_window
        return np.round(np.clip(lasers, self.min_distance_seen, self.max_distance_obstacle), self.sensitivity)

    def find_food(self):
        return self.compute_distance_food(), self.compute_angle_food()

    def compute_angle_food(self):
        vector_food_in_frame_robot = self.rotate_vector_2D(np.array([self.food_x - self.robot_x, self.food_y - self.robot_y]), -self.robot_theta * 180 / np.pi)
        return self.normalize_angle(round(np.arctan2(vector_food_in_frame_robot[1], vector_food_in_frame_robot[0]), self.sensitivity))
        
    def compute_distance_food(self):
        return round(np.linalg.norm(np.array([self.robot_x, self.robot_y]) - np.array([self.food_x, self.food_y])), self.sensitivity)

    @staticmethod
    def normalize_angle(angle):
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < - np.pi:
            angle = angle + 2 * np.pi
        return angle

    @staticmethod
    def rotate_vector_2D(vector, angle_degrees):
        angle_rad = angle_degrees * TO_RAD
        sin_, cos_ = sin(angle_rad), cos(angle_rad)
        return np.array([[cos_, -sin_],[sin_, cos_]]).dot(vector)

