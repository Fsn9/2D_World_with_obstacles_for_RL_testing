# Auxiliar libraries
from numpy.random import rand, randint
from numpy import pi, array
from math import sqrt
import matplotlib.pyplot as plt
from collections import deque
from random import sample, choice
# Auxiliar classes
from utils import Integrator, Exponential
from observation_processor import ObservationProcessor
from reward_function import Rewarder
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
# DQN utils
from dqn_utils import *
# Tensorboard
from torch.utils.tensorboard import SummaryWriter # Tensorboard
# JSON
import json
# OS
from os import getcwd
# ROS
import rospy
import rospkg

TO_DEG = 57.29577
TO_RAD = 0.01745

class DQN(object):
    def __init__(self):
        # Path
        rospack = rospkg.RosPack()
        self.path = rospack.get_path('2D_World_with_obstacles')

        # Hyperparameters load
        with open(self.path + '/src/hyperparameters.json','r') as hp_file:
            self.hp = json.load(hp_file)

        # DQN parameters
        self.episodes = self.hp['dqn']['episodes']
        self.actual_episodes = self.episodes
        self.initial_epsilon = self.hp['dqn']['initial_epsilon']
        self.initial_gamma = self.hp['dqn']['initial_gamma']
        self.learning_rate = self.hp['dqn']['learning_rate']
        self.momentum = self.hp['dqn']['momentum']
        self.actual_gamma = self.initial_gamma
        self.actual_epsilon = self.initial_epsilon
        self.is_learning = True
        self.reward = 0.0

        # Constants
        self.max_distance_map = sqrt(self.hp['map']['width'] ** 2 + self.hp['map']['height'] ** 2)
        self.max_angular_velocity = self.hp['robot']['max_angular_velocity']
        self.num_actions = self.hp['control']['num_actions']
        self.linear_velocity = self.hp['control']['linear_velocity']
        self.phi = self.hp['control']['phi']
        self.actual_max_angular_velocity = self.max_angular_velocity * self.phi
        self.reset_duration = self.hp['control']['reset_duration']
        self.sample_rate = self.hp['control']['sample_rate']
        self.max_distance_seen = self.hp['control']['max_distance_seen']
        self.min_distance_seen = self.hp['control']['min_distance_seen']
        self.frame_period = self.hp['dqn']['frame_distance'] / self.linear_velocity
        self.window_collision = self.hp['dqn']['window_collision']
        self.just_reseted = False
        
        # Action space constraint check
        min_diameter = 2.0 * self.linear_velocity / self.actual_max_angular_velocity
        if min_diameter <= self.min_distance_seen:
            raise Exception('Minimum diameter possible to be covered by the robot of {} is lower than the minimum distance of {}'.format(min_diameter, self.min_distance_seen))
        self.action_space = [round(i * self.actual_max_angular_velocity / int(self.num_actions * 0.5), 4) for i in range(-int(self.num_actions * 0.5), int(self.num_actions * 0.5) + 1)]

        # Auxiliary variables
        self.distance_to_goal_tolerance = self.hp['control']['min_distance_seen']
        self.reset_goal_x = 0.0
        self.reset_goal_y = 0.0

        # Statistics
        #self.path = '~/FastTurtle/'
        self.path = rospack.get_path('2D_World_with_obstacles')
        self.writer = SummaryWriter(self.path + "/src/log_tensorboard")
        self.collected_rewards_per_episode = list()
        self.collected_losses_per_episode = list()
        self.total_reward = 0.0
        self.total_loss = 0.0
        self.global_step = 0
        self.steps = 0

        # Create object of type Observer to observe laser and goal position
        self.observation_processor = ObservationProcessor(hyperparameters = self.hp)

        # Initialize reward function
        self.rewarder = Rewarder(hyperparameters = self.hp)

        # Initialize replay memory, state info and NN parameters
        self.batch_size = self.hp['dqn']['batch_size']
        self.memory_capacity = self.hp['dqn']['replay_memory_capacity']
        self.number_of_frames = self.hp['dqn']['number_of_frames']
        self.replay_memory = ReplayMemory(self.memory_capacity, self.batch_size)
        self.frame_buffer = FrameBuffer(self.number_of_frames)
        self.target_network_update_period = self.hp['dqn']['target_network_update_period']

        # Networks
        self.input_dimension = self.hp['control']['angle_window']
        self.policy_network = DCNN1D(self.input_dimension, self.number_of_frames, self.num_actions, self.learning_rate, self.momentum)
        self.target_network = DCNN1D(self.input_dimension, self.number_of_frames, self.num_actions, self.learning_rate, self.momentum)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval() # No training in target network, only predictions

        # Auxiliar variables
        self.action = None
        self.action_index = None
        self.observing = True
        self.episode_started = True
        self.steps = 0
        self.current_laser_tensor, self.next_laser_tensor, self.current_goal_tensor, self.next_goal_tensor = None, None, None, None


    def normalize(self, tensor, max_, min_):
        return (tensor - min_) / (max_ - min_)

    def prepare_batch(self, tensors):
        return torch.stack(tuple(tensors))

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

    def its_time_to_update_target_network(self):
        #episode_refresh_rate = int(round(self.target_network_update_rate(self.episodes - self.actual_episodes)))
        #print('rate:',self.first_episode_refresh_period / episode_refresh_rate, '. episodes:',self.actual_episodes)
        #return (self.episodes - self.actual_episodes) % (self.first_episode_refresh_period / episode_refresh_rate) == 0.0
        #return self.actual_episodes % self.target_network_update_period == 0
        return self.global_step % self.target_network_update_period == 0

    def its_time_to_reset_replay_memory(self):
        return False

    def deactivate_learning(self):
        self.is_learning = False

    def decay_hyper_parameters(self):
        self.actual_episodes -= 1
        self.actual_epsilon = -(self.initial_epsilon / self.episodes) * (self.episodes - self.actual_episodes) + self.initial_epsilon
        self.actual_gamma = -(self.initial_gamma / self.episodes) * (self.episodes - self.actual_episodes) + self.initial_gamma

    # Good
    def save_stats(self):
        self.writer.add_scalar('Total loss by episode', self.total_loss, self.episodes - self.actual_episodes)
        self.writer.add_scalar('Total reward by episode', self.total_reward, self.episodes - self.actual_episodes)
        self.writer.add_scalar('Total steps by episode', self.steps, self.episodes - self.actual_episodes)
        try:
            self.writer.add_scalar('Moving average reward', sum(self.collected_rewards_per_episode) / len(self.collected_rewards_per_episode), self.episodes - self.actual_episodes)
            self.writer.add_scalar('Moving TD-error(loss)', sum(self.collected_losses_per_episode) / len(self.collected_losses_per_episode), self.episodes - self.actual_episodes)
        except ZeroDivisionError:
            print('Division by zero. Passing')
            self.writer.add_scalar('Average reward', 0.0, self.episodes - self.actual_episodes)

        del self.collected_rewards_per_episode[:]
        del self.collected_losses_per_episode[:]

    def observe(self):
        lasers, polar_goal = self.observation_processor.process(self.lasers, self.goal_x, self.goal_y, self.robot_x, self.robot_y, self.robot_theta)
        return Observation(lasers, polar_goal)

    def normalize(self, tensor, max_, min_):
        return (tensor - min_) / (max_ - min_)

    def denormalize(self, tensor, max_, min_):
        return tensor * (max_ - min_) + min_

    def pre_process_observations(self):
        frames = self.frame_buffer.load()
        laser_tensors = list()
        goal_tensors = list()

        # Process data in frames and transform in tensors
        for i in range(self.number_of_frames):
            laser_values = self.observation_processor.process_laser_data(frames[i][0])
            goal_values = self.observation_processor.process_goal_data(frames[i][1][0], frames[i][1][1], frames[i][1][2], frames[i][1][3], frames[i][1][4])
            laser_tensors.append(torch.FloatTensor(laser_values))
            goal_tensors.append(torch.FloatTensor(goal_values))
        # laser
        laser_tensor = torch.stack(tuple(laser_tensors), dim = 0).unsqueeze(0)
        laser_tensor = self.normalize(laser_tensor, self.max_distance_seen, self.min_distance_seen)
        
        # goal
        goal_tensor = torch.stack(tuple(goal_tensors), dim = 0)
        distance_goal_tensor = goal_tensor[:,0]
        angle_goal_tensor = goal_tensor[:,1]
        distance_goal_tensor = self.normalize(distance_goal_tensor, self.max_distance_map, self.min_distance_seen).view(-1,1)
        angle_goal_tensor = self.normalize(angle_goal_tensor, pi, -pi).view(-1,1)
        goal_tensor = torch.cat((distance_goal_tensor, angle_goal_tensor), dim = 1).unsqueeze(0)
        return laser_tensor, goal_tensor

    def decide(self, input_tensors):
        self.policy_network.eval()
        current_laser_tensor, current_goal_tensor = input_tensors
        with torch.no_grad(): 
            policy_q_actions = self.policy_network(current_laser_tensor, current_goal_tensor)
            action_index = policy_q_actions.argmax().item()
            action = self.action_space[action_index]
        self.policy_network.train()
        return action_index, action

    def e_greedy_decision(self, input_tensors):
        if rand() > self.actual_epsilon: # exploit
            action_index, action = self.decide(input_tensors) 
        else: # explore
            action_index = randint(0, self.num_actions)
            action = self.action_space[action_index]
        self.robot_action = action # extra
        return action_index, action

    def process_minibatch(self):
        minibatch = self.replay_memory.sample()  
        with torch.no_grad(): 
            max_q_targets_next_state = self.target_network(minibatch.next_lasers, minibatch.next_polar_goals).squeeze(0).detach().max(1)[0].unsqueeze(-1)
        q_predictions = self.policy_network(minibatch.current_lasers, minibatch.current_polar_goals).squeeze(0).gather(1, minibatch.actions)
        rewards = minibatch.rewards
        terminals = minibatch.terminals
        return q_predictions, max_q_targets_next_state, rewards, terminals

    def compute_loss(self, q_predictions, q_targets):
        return self.policy_network.criterion(q_predictions, q_targets)

    def learn(self, loss):
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm = 1.0) # gradient clipping
        self.policy_network.optimizer.step()

    def run(self, observation, last_action, terminal):
        print('steps:',self.steps)
        # Initial observation in the beggining of the episode
        if self.episode_started:
            self.frame_buffer.add(observation)
            if len(self.frame_buffer) == self.number_of_frames:
                self.observing = False
                self.episode_started = False
            return last_action
        # While on the episode
        else:
            # Decide
            if not self.observing and not terminal[0]:
                # Pre process current observation if frame buffer is empty
                if len(self.frame_buffer) != 0:
                    self.current_laser_tensor, self.current_goal_tensor = self.pre_process_observations()
                # Decide action
                self.action_index, self.action = self.e_greedy_decision((self.current_laser_tensor, self.current_goal_tensor))

                # will observe the effects of the action
                self.observing = True
                return self.action
            # If done deciding, observe the next state or learn
            else:
                # If terminal, next frames are all equal and terminal
                if terminal[0]:
                    for i in range(self.number_of_frames):
                        self.frame_buffer.add(observation)
                # Observe the next state
                else:
                    self.frame_buffer.add(observation)
                # Check if collecting is done
                if len(self.frame_buffer) == self.number_of_frames:
                    self.observing = False
                # If still collecting mantain the same action and return
                else:
                    return last_action

                # Check terminal  and learn
                self.next_laser_tensor, self.next_goal_tensor = self.pre_process_observations()
                distance_goal = self.denormalize(self.next_goal_tensor[0,:,0], self.max_distance_map, self.min_distance_seen).mean()
                angle_goal = self.denormalize(self.next_goal_tensor[0,:,1], pi, -pi).mean()
                reward = self.rewarder.compute_reward(distance_goal, angle_goal, terminal[1])
                self.collected_rewards_per_episode.append(reward)
                self.total_reward += reward
                
                print('REWARD: ',reward,'\n')

                # Save transition in replay memory
                self.replay_memory.store(Transition((self.current_laser_tensor, self.current_goal_tensor), self.action_index, reward, (self.next_laser_tensor, self.next_goal_tensor), terminal[0]))

                # Current state <- Next state
                self.current_laser_tensor, self.current_goal_tensor = self.next_laser_tensor.clone(), self.next_goal_tensor.clone()

                # If enough experiences
                if len(self.replay_memory) < self.batch_size:
                    return last_action

                # Only learn when replay memory has size N (change)
                # Process minibatch
                q_predictions, max_q_targets_next_state, rewards, terminals = self.process_minibatch()

                # Compute q_targets
                q_targets = rewards + self.actual_gamma * max_q_targets_next_state * (1 - terminals)
                
                # Loss calculation
                loss = self.compute_loss(q_predictions, q_targets)
                self.collected_losses_per_episode.append(loss.item())
                self.total_loss += loss.item()

                # Learn (backward pass)
                self.learn(loss)

                # Increment number of steps
                self.global_step += 1 # per learning cycle
                self.steps += 1 # per episode

                # Update target network
                if self.its_time_to_update_target_network():
                    self.update_target_network()

                if terminal[0]:
                    self.decay_hyper_parameters()
                    self.save_stats()
                    self.steps = 0
                    self.total_reward = 0.0
                    self.total_loss = 0.0
                    self.episode_started = True
                    self.observing = True
                    self.frame_buffer.clear()
                return self.action