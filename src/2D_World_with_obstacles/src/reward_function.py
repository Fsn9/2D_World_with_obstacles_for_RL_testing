#!/usr/bin/env python

import rospy
from numpy import array
from numpy.linalg import norm
#from geometry_msgs.msg import Pose2D
#from std_msgs.msg import Float64
#from nav_msgs.msg import OccupancyGrid
#from qlearning_training_simulation.msg import FeatureStateVector
from math import pi, sqrt
from utils import Ramp, Exponential, StateSpace

class Rewarder:
    def __init__(self, hyperparameters):
        self.hp = hyperparameters

        # Robot info
        self.robot_radius = self.hp['robot']['radius']

        # Parameters
        self.min_distance_goal = self.hp['control']['min_distance_seen']
        self.max_distance_goal = sqrt(self.hp['map']['width']**2 + self.hp['map']['height']**2)
        self.distance_goal_resolution = self.hp['state']['distance_obstacle_resolution']
        self.max_angle_goal = self.hp['state']['max_angle_goal']
        self.min_angle_goal = self.hp['state']['min_angle_goal']
        self.angle_goal_resolution = self.hp['state']['angle_goal_resolution']
        self.min_distance_obstacle = self.hp['control']['min_distance_seen']
        self.max_distance_seen = self.hp['control']['max_distance_seen'] 
        self.alpha = self.hp['state']['alpha'] # percentage of maximum distance seen by the robot
        self.max_distance_obstacle = self.alpha * self.max_distance_seen # actual max distance seen by the robot
        self.distance_obstacle_resolution = self.hp['state']['distance_obstacle_resolution'] # resolution of distance to obstacle
        self.angle_window_deg = self.hp['control']['angle_window'] # angle_window seen by the robot
        self.angle_window = self.angle_window_deg * pi / 180
        self.angle_obstacle_resolution = self.hp['state']['angle_obstacle_resolution'] # resolution of angle to obstacle
        
        '''goal'''
        # distance  
        self.critical_distance_goal = self.hp['reward_function']['critical_distance_goal']
        self.goal_big_prize = self.hp['reward_function']['goal_prize']
        self.max_reward_distance_goal = self.hp['reward_function']['max_reward_distance_goal']
        self.min_reward_distance_goal = self.hp['reward_function']['min_reward_distance_goal']
        self.reward_distance_goal = Ramp(y2 = self.max_reward_distance_goal, y1 = self.min_reward_distance_goal, x2 = self.min_distance_goal, x1 =  self.max_distance_goal)

        # angle
        self.critical_angle_goal = self.hp['reward_function']['critical_angle_goal'] # degrees
        self.max_reward_angle_goal = self.hp['reward_function']['max_reward_angle_goal']
        self.min_reward_angle_goal = self.hp['reward_function']['min_reward_angle_goal']
        self.beta = self.hp['reward_function']['beta'] # reward value factor from where the steering angle to the goal starts to become acceptable
        self.acceptable_reward_value = self.min_reward_angle_goal / self.beta
        self.reward_angle_goal_exp1 = Exponential(y2 = self.min_reward_angle_goal, y1 = self.acceptable_reward_value, x2 = -180 , x1 = -self.critical_angle_goal)
        self.reward_angle_goal1 = Ramp(y2 = self.acceptable_reward_value, y1 = self.max_reward_angle_goal, x2 = -self.critical_angle_goal, x1 = 0.0)
        self.reward_angle_goal2 = Ramp(y2 = self.max_reward_angle_goal, y1 = self.acceptable_reward_value, x2 = 0.0, x1 = self.critical_angle_goal)
        self.reward_angle_goal_exp2 = Exponential(y2 = self.acceptable_reward_value, y1 = self.min_reward_angle_goal, x2 = self.critical_angle_goal, x1 = 180)
        self.reward_angle_goal = Ramp(y2 = self.max_reward_angle_goal, y1 = self.min_reward_angle_goal, x2 = 0.0, x1 = pi)

        '''OBSTACLE'''
        # distance
        self.critical_distance_obstacle = self.hp['reward_function']['critical_distance_obstacle']
        self.max_reward_distance_obstacle = self.hp['reward_function']['max_reward_distance_obstacle']
        self.min_reward_distance_obstacle = self.hp['reward_function']['min_reward_distance_obstacle']
        self.collision_penalty = self.hp['reward_function']['collision_penalty']
        self.reward_distance_obstacle = Ramp(y2 = self.min_reward_distance_obstacle, y1 = self.max_reward_distance_obstacle, x2 = self.min_distance_obstacle, x1 = self.critical_distance_obstacle)

        # angle
        self.critical_angle_obstacle = self.hp['reward_function']['critical_angle_obstacle']
        self.max_reward_angle_obstacle = self.hp['reward_function']['max_reward_angle_obstacle']
        self.min_reward_angle_obstacle = self.hp['reward_function']['min_reward_angle_obstacle']
        self.reward_angle_obstacle1 = Ramp(y2 = self.max_reward_angle_obstacle, y1 = self.min_reward_angle_obstacle, x2 = -self.critical_angle_obstacle, x1 = 0.0)
        self.reward_angle_obstacle2 = Ramp(y2 = self.min_reward_angle_obstacle, y1 = self.max_reward_angle_obstacle, x2 = 0.0, x1 = self.critical_angle_obstacle)

    def compute_reward(self, distance_goal, angle_goal, terminal):
        # terminal state
        if terminal == 'collision':
            return self.collision_penalty
        # reached goal
        elif terminal == 'goal':
            return self.goal_big_prize
        else:
            print('rd: ',self.reward_distance_goal(distance_goal))
            print('ra: ',self.reward_angle_goal(abs(angle_goal)))
            #return self.reward_distance_goal(distance_goal) + 0.25 * self.reward_angle_goal(abs(angle_goal))
            return self.reward_angle_goal(abs(angle_goal))
