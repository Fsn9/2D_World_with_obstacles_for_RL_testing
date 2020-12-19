# Auxiliar libraries
from numpy.random import rand, randint
from numpy import pi, array
from math import sqrt
import matplotlib.pyplot as plt
from collections import deque
from random import sample
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

TO_DEG = 57.29577
TO_RAD = 0.01745

class DQN(object):
    def __init__(self, first_laser_scan, first_goal_position, robot_pose):
        # Hyperparameters
        with open('hyperparameters.json','r') as hp_file:
            hp_stream = hp_file.read()
        hp = json.load(hp_stream)

        # DQN parameters
        self.episodes = hp['dqn']['episodes']
        self.initial_epsilon = hp['dqn']['initial_epsilon']
        self.initial_gamma = hp['dqn']['initial_gamma']
        self.learning_rate = hp['dqn']['learning_rate']
        self.actual_episodes = self.episodes
        self.actual_gamma = self.initial_gamma
        self.actual_epsilon = self.initial_epsilon
        self.is_learning = True
        self.reward = 0.0

        # Constants
        self.max_distance_map = sqrt(hp['map']['width'] ** 2 + hp['map']['height'] ** 2)
        self.max_angular_velocity = hp['robot']['max_angular_velocity']
        self.num_actions = hp['control']['num_actions']
        self.linear_velocity = hp['control']['linear_velocity']
        self.phi = hp['control']['phi']
        self.actual_max_angular_velocity = self.max_angular_velocity * self.phi
        self.max_distance_per_episode = hp['dqn']['max_distance_per_episode']
        self.reset_duration = hp['control']['reset_duration']
        self.sample_rate = hp['dqn']['sample_rate']
        self.max_distance_seen = hp['dqn']['max_distance_seen']
        self.min_distance_seen = hp['dqn']['min_distance_seen']
        self.frame_period = hp['dqn']['frame_distance'] / self.linear_velocity
        self.window_collision = hp['dqn']['window_collision']
        self.just_reseted = False
        
        # Action space constraint check
        min_diameter = 2.0 * self.linear_velocity / self.actual_max_angular_velocity
        if min_diameter <= self.min_distance_seen:
            raise Exception('Minimum diameter possible to be covered by the robot of {} is lower than the minimum distance of {}'.format(min_diameter, self.min_distance_seen))
        self.action_space = [round(i * self.actual_max_angular_velocity / int(self.num_actions * 0.5), 4) for i in range(-int(self.num_actions * 0.5), int(self.num_actions * 0.5) + 1)]

        # Auxiliary variables
        self.current_state = ()
        self.robot_pose = array([robot_pose.x, robot_pose.y, robot_pose.theta])
        self.robot_x, self.robot_y, self.robot_theta = robot_pose.x, robot_pose.y, robot_pose.theta
        self.laser_scan = first_laser_scan
        self.lasers = [laser if laser <= self.laser_scan.range_max else self.laser_scan.range_max for laser in self.laser_scan.ranges]
        self.goal_x, self.goal_y = first_goal_position.x, first_goal_position.y
        self.robot_x = self.robot_pose[0]
        self.robot_y = self.robot_pose[1]
        self.last_position = array([robot_pose.x, robot_pose.y])
        self.integrator_tolerance = 250
        self.is_stuck_distance_tolerance = self.linear_velocity * (1.0 / self.sample_rate) * 0.05
        self.integrator = Integrator(capacity = self.integrator_tolerance)
        self.reset_distance_tolerance = 0.2
        self.max_angular_velocity_reset = 0.4
        self.distance_to_goal_tolerance = hp['control']['min_distance_seen']
        self.reset_goal_x = 0.0
        self.reset_goal_y = 0.0

        # Statistics
        #self.path = '~/FastTurtle/'
        #self.writer = SummaryWriter(self.path + "/src/log_tensorboard")
        #self.writer = None
        #self.collected_rewards_per_episode = list()

        # Create object of type Observer to observe laser and goal position
        self.observation_processor = ObservationProcessor()

        # Initialize reward function
        self.rewarder = Rewarder()

        # Initialize replay memory, state info and NN parameters
        self.batch_size = hp['dqn']['batch_size']
        self.memory_capacity = hp['dqn']['replay_memory_capacity']
        self.number_of_frames = hp['dqn']['number_of_frames']
        self.replay_memory = ReplayMemory(self.memory_capacity, self.batch_size)
        self.frame_buffer = FrameBuffer(self.number_of_frames)
        self.target_network_first_update_period_percentage = hp['dqn']['target_network_first_update_period_percentage']
        self.first_episode_refresh_period =  self.target_network_first_update_period_percentage * self.episodes
        self.max_target_network_update_rate = rospy.get_param('/max_target_network_update_rate')
        self.target_network_update_rate = Exponential(x2 = self.episodes, x1 = 0.0, y2 = self.max_target_network_update_rate, y1 = 1.0)
        
        # Networks
        self.input_dimension = hp['state']['angle_window']
        self.policy_network = DCNN1D(self.input_dimension, self.number_of_frames, self.num_actions, self.learning_rate)
        self.target_network = DCNN1D(self.input_dimension, self.number_of_frames, self.num_actions, self.learning_rate)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval() # No training in target network, only predictions

        # Robot brain info
        self.robot_state = []
        self.robot_action = 0.0

    def frames_to_tensors(self, frames):
        laser_tensors = list()
        goal_tensors = list()

        # Get values from message
        for i in range(self.number_of_frames):
            laser_values = frames[i].lasers
            goal_values = frames[i].polar_goal
            laser_tensors.append(torch.FloatTensor(laser_values))
            goal_tensors.append(torch.FloatTensor(goal_values))

        # Preprocess (stack and normalize)
        # lasers
        laser_tensor = torch.stack(tuple(laser_tensors), dim = 0).unsqueeze(0)
        laser_tensor = self.normalize(laser_tensor, self.max_distance_seen, self.min_distance_seen)
        
        # goal
        goal_tensor = torch.stack(tuple(goal_tensors), dim = 0)
        distance_goal_tensor = goal_tensor[:,0]
        angle_goal_tensor = goal_tensor[:,1]
        distance_goal_tensor = self.normalize(distance_goal_tensor, self.max_distance_map, 0.0).view(-1,1)
        angle_goal_tensor = self.normalize(angle_goal_tensor, pi, -pi).view(-1,1)
        goal_tensor = torch.cat((distance_goal_tensor, angle_goal_tensor), dim = 1).unsqueeze(0)
        return laser_tensor, goal_tensor

    def normalize(self, tensor, max_, min_):
        return (tensor - min_) / (max_ - min_)

    def prepare_batch(self, tensors):
        return torch.stack(tuple(tensors))

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

    def its_time_to_update_target_network(self):
        episode_refresh_rate = int(round(self.target_network_update_rate(self.episodes - self.actual_episodes)))
        print 'rate:',self.first_episode_refresh_period / episode_refresh_rate, '. episodes:',self.actual_episodes
        return (self.episodes - self.actual_episodes) % (self.first_episode_refresh_period / episode_refresh_rate) == 0.0

    def its_time_to_reset_replay_memory(self):
        return False

    def deactivate_learning(self):
        self.is_learning = False

    def check_end_episode(self):
        if self.ate_goal(): # Check if ate goal
            self.reset_goal_position()
            return True ,'ate_goal'

        self.increase_distance() # Integrate distance to keep track of stuck situations

        self.episode_is_over, cause = self.is_episode_over()
        
        # If episode ended:
        if self.episode_is_over:
            self.actual_episodes -= 1
            # Routine of going back
            start_time = rospy.get_rostime()
            while rospy.get_rostime() - start_time < self.reset_duration:
                self.go_back()
            self.reset_integrator()
            self.save_last_position()
            self.decay_hyper_parameters() # Decay hyper parameters
            #try:
            #    self.writer.add_scalar('Average reward', sum(self.collected_rewards_per_episode)/len(self.collected_rewards_per_episode), self.episodes - self.actual_episodes)
            #except ZeroDivisionError:
            #    print('Division by zero. Passing')
            #    pass
            del self.collected_rewards_per_episode[:]
            return self.episode_is_over, cause
        
        if self.actual_episodes == 0: # If actual episode counter ended, finish learning
            self.deactivate_learning()
            return True, 'end'
        else:
            return False, None

    def decay_hyper_parameters(self):
        self.actual_episodes -= 1
        self.actual_epsilon = -(self.initial_epsilon / self.episodes) * (self.episodes - self.actual_episodes) + self.initial_epsilon
        self.actual_gamma = -(self.initial_gamma / self.episodes) * (self.episodes - self.actual_episodes) + self.initial_gamma

    def reset_episode(self):
        # Routine of going back
        self.stop_robot()
        start_time = rospy.get_rostime()
        while rospy.get_rostime() - start_time < self.reset_duration:
            self.go_back()       

    def go_back(self):
        self.action_decided.linear.x, self.action_decided.angular.z = -self.linear_velocity, self.max_angular_velocity_reset * rand() 
        self.action_publisher.publish(self.action_decided)

    def ate_goal(self, current_frames, next_frames):
        recent_current = current_frames[self.number_of_frames - 1][0]
        oldest_next = next_frames[0][0]
        if (recent_current + oldest_next) * 0.5 < self.distance_to_goal_tolerance:
            self.reset_goal_position()
            return True
        return False

    def is_episode_over(self):
        # if robot is stuck, reset episode
        if self.robot_stuck():
            self.stop_robot() # Stop robot
            self.current_state = self.feature_state_vector_to_tuple(self.observe()) # Observe new state
            return True, 'collision'
        else:
            return False, None

    def reset_integrator(self):
        self.integrator.reset()

    def increase_distance(self):
        self.integrator.update(x = self.robot_x, y = self.robot_y)

    def save_last_position(self):
        self.last_position = array([self.robot_x, self.robot_y])

    def robot_stuck(self):    
        return self.integrator() < self.is_stuck_distance_tolerance

    def stop_robot(self):
        self.action_decided.linear.x, self.action_decided.angular.z = 0.0, 0.0  
        self.action_publisher.publish(self.action_decided)

    def reset_goal_position(self):
        rospy.wait_for_service('reset_goal_position')
        try:
            reset_goal_position_request = rospy.ServiceProxy('reset_goal_position', ResetFoodPosition)
            response = reset_goal_position_request()
            self.reset_goal_x = response.x
            self.reset_goal_y = response.y
            return response
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)        
   
    def check_terminal(self, laser_tensor):
        if self.just_reseted:
            self.just_reseted = False
            return False
        average_color = 0.0
        averages = list()
        frames = laser_tensor.clone().squeeze(0)
        center = self.input_dimension // 2
        variation = 0.0
        for idx in range(self.number_of_frames):
            if (idx + 1) == self.number_of_frames:
                break
            averages.append(abs((frames[idx + 1][center - self.window_collision:center + self.window_collision] - frames[idx][center - self.window_collision:center + self.window_collision]).mean().item()))
            variation += (frames[idx + 1][center - self.window_collision:center + self.window_collision] - frames[idx][center - self.window_collision:center + self.window_collision]).abs().sum()
        average_color = sum(averages) / len(averages)
        print 'variation:', variation.item()
        print 'average_color:', average_color
        terminal = variation < 0.8 and average_color < 0.001
        if terminal:
            self.just_reseted = True
        return terminal

    def observe(self):
        lasers, polar_goal = self.observation_processor.process(self.lasers, self.goal_x, self.goal_y, self.robot_x, self.robot_y, self.robot_theta)
        return Observation(lasers, polar_goal)

    def collect_frames(self):
        for i in range(self.number_of_frames):
            self.frame_buffer.add(self.observe())
            rospy.sleep(self.frame_period) # melhorar
        return self.frame_buffer.get_data()

    def get_observation_data(self):
        current_laser_frames, current_polar_goal_frames = self.collect_frames() # Collects state frames
        current_laser_tensor, current_polar_goal_tensor = self.frames_to_tensors(self.frame_buffer.load()) # frames to tensors
        self.robot_state = 'lasers: '+str(current_laser_frames)+',polar_goal: '+str(current_polar_goal_frames) # extra
        return current_laser_frames, current_polar_goal_frames, current_laser_tensor, current_polar_goal_tensor

    def decide(self, input_tensors):
        self.policy_network.eval()
        current_laser_tensor, current_polar_goal_tensor = input_tensors
        with torch.no_grad(): 
            policy_q_actions = self.policy_network(current_laser_tensor, current_polar_goal_tensor)
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

    def act(self, action):
        self.action_decided.linear.x = self.linear_velocity
        self.action_decided.angular.z = action
        self.action_publisher.publish(self.action_decided)

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

    def run(self):
        # Observe current state
        current_laser_frames, current_polar_goal_frames, current_laser_tensor, current_polar_goal_tensor = self.get_observation_data()

        # Decide action
        action_index, action = self.e_greedy_decision((current_laser_tensor, current_polar_goal_tensor))
           
        # Act
        self.act(action)

        # Observe next state
        next_laser_frames, next_polar_goal_frames, next_laser_tensor, next_polar_goal_tensor = self.get_observation_data()

        # Check if robot collided or ate goal
        terminal = self.check_terminal(next_laser_tensor)
        ate_goal = self.ate_goal(current_polar_goal_frames, next_polar_goal_frames)

        if terminal: print 'collided...'
        if ate_goal: print 'Ate_goal!'

        # Reward
        reward = self.rewarder.compute_reward(next_laser_frames, next_polar_goal_frames, terminal, ate_goal)
        print 'reward: ',reward,'\n'

        # Save transition in replay memory
        self.replay_memory.store(Transition((current_laser_tensor, current_polar_goal_tensor), action_index, reward, (next_laser_tensor, next_polar_goal_tensor), terminal))

        # If enough experiences
        if len(self.replay_memory) < self.batch_size:
            return
            
        # Only learn when replay memory has size N (change)
        # Process minibatch
        q_predictions, max_q_targets_next_state, rewards, terminals = self.process_minibatch()

        # Compute q_targets
        q_targets = rewards + self.actual_gamma * max_q_targets_next_state * (1 - terminals)
        
        # Loss calculation
        loss = self.compute_loss(q_predictions, q_targets)

        # Learn (backward pass)
        self.learn(loss)

        # Update target network
        if self.its_time_to_update_target_network():
            self.update_target_network()

        #if self.its_time_to_reset_replay_memory():
        #    self.replay_memory.reset()

        if terminal:
            self.reset_episode()
            self.decay_hyper_parameters()

        self.frame_buffer.clear()

class DCNN1D(torch.nn.Module):
    def __init__(self, input_dim, input_channels, output_dim, learning_rate):
        super(DCNN1D, self).__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_frames = rospy.get_param('/number_of_frames')
        self.goal_dimension = self.number_of_frames * 2
        self.laser_dimension = self.number_of_frames * self.input_dim

        # convolutional layers 
        self.conv1 = nn.Conv1d(in_channels = self.input_channels, out_channels = 8, kernel_size = 5, stride = 2, bias = False)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, bias = False)
        self.bn2 = nn.BatchNorm1d(16)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride = 2)

        # fully connected layer
        self.flatten_dim = self.conv1d_size(self.input_dim, 5, 2)
        self.flatten_dim = self.flatten_dim // 2
        self.flatten_dim = self.conv1d_size(self.flatten_dim, 3, 2)  
        self.flatten_dim = self.flatten_dim // 2
        self.flatten_dim = self.flatten_dim * 16
        self.linear1 = nn.Linear(in_features = self.flatten_dim + self.goal_dimension, out_features = 128)
        self.linear2 = nn.Linear(in_features = 128, out_features = self.output_dim)

        # criterion, optimizer, learning_rate
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss() # change to huber loss
        self.optimizer = optimizers.RMSprop(self.parameters(), lr = learning_rate)

    def forward(self, laser_tensor, goal_tensor):
        # Pass laser first through conv layers
        laser_tensor = self.conv1(laser_tensor)
        laser_tensor = self.minpool(laser_tensor)
        laser_tensor = F.relu(self.bn1(laser_tensor))
        laser_tensor = self.conv2(laser_tensor)
        laser_tensor = self.minpool(laser_tensor)
        laser_tensor = F.relu(self.bn2(laser_tensor))

        # Flatten tensors
        laser_tensor = torch.flatten(laser_tensor, start_dim = 1)
        goal_tensor = torch.flatten(goal_tensor, start_dim = 1)

        # Concatenate goal_tensor
        final_tensor = torch.cat((laser_tensor, goal_tensor), dim = 1).unsqueeze(0)

        # Concat with goal tensor and pass both in fully connected layer
        final_tensor = self.linear1(final_tensor)
        final_tensor = self.linear2(final_tensor)
        return final_tensor #softmax?

    def minpool(self, tensor):
        return -self.maxpool(-tensor)

    def conv1d_size(self, size, kernel_size, stride):
        return 1 + (size - (kernel_size - 1) - 1) // stride

class Minibatch(object):
    CURRENT_STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    TERMINAL = 4
    LASER = 0
    POLAR_FOOD = 1
    def __init__(self, minibatch):
        self.batch_size = len(minibatch)
        self.number_of_frames = rospy.get_param('/number_of_frames')
        # Lists to store individual elements
        self.current_lasers_batch_list = list()
        self.current_polar_goal_batch_list = list()
        self.next_lasers_batch_list = list()
        self.next_polar_goal_batch_list = list()
        self.action_batch_list = list()
        self.reward_batch_list = list()
        self.terminal_batch_list = list()

        for idx, transition in enumerate(minibatch):
            self.current_lasers_batch_list.append((transition.current_laser))
            self.current_polar_goal_batch_list.append((transition.current_polar_goal))
            self.next_lasers_batch_list.append((transition.next_laser))
            self.next_polar_goal_batch_list.append((transition.next_polar_goal))
            self.action_batch_list.append(transition.action)
            self.reward_batch_list.append(transition.reward)
            self.terminal_batch_list.append(transition.terminal)

        self._current_lasers = torch.cat(tuple(self.current_lasers_batch_list), dim = 0)
        self._current_polar_goals = torch.cat(tuple(self.current_polar_goal_batch_list), dim = 0)
        self._next_lasers = torch.cat(tuple(self.next_lasers_batch_list), dim = 0)
        self._next_polar_goals = torch.cat(tuple(self.next_polar_goal_batch_list), dim = 0)
        self._actions = torch.tensor(self.action_batch_list).view(self.batch_size, 1)
        self._rewards = torch.tensor(self.reward_batch_list).view(self.batch_size, 1)
        self._terminals = torch.tensor(self.terminal_batch_list).byte().view(self.batch_size, 1)
              
    @property
    def current_lasers(self):
        return self._current_lasers
    @property
    def current_polar_goals(self):
        return self._current_polar_goals
    @property
    def next_lasers(self):
        return self._next_lasers
    @property
    def next_polar_goals(self):
        return self._next_polar_goals
    @property
    def actions(self):
        return self._actions
    @property
    def rewards(self):
        return self._rewards
    @property
    def terminals(self):
        return self._terminals
            
class Transition(object):
    def __init__(self, current_state, action, reward, next_state, terminal):
        self._current_state = current_state
        self._action = action
        self._reward = reward
        self._next_state = next_state
        self._terminal = terminal
    @property
    def current_state(self):
        return self._current_state
    @property
    def action(self):
        return self._action
    @property
    def reward(self):
        return self._reward
    @property
    def next_state(self):
        return self._next_state
    @property
    def terminal(self):
        return self._terminal
    @property
    def current_laser(self):
        return self._current_state[0]
    @property
    def next_laser(self):
        return self._next_state[0]
    @property
    def current_polar_goal(self):
        return self._current_state[1]
    @property
    def next_polar_goal(self):
        return self._next_state[1]
    
class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen = self.capacity)

    def store(self, transition):
        self.memory.append(transition)

    def reset(self):
        del self.memory[:]
    
    def sample(self):
        return Minibatch(sample(self.memory, self.batch_size)) # sample and delete. CHANGE

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return 'Replay memory with a current length of ' + str(self.__len__()) + 'transitions'

class FrameBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = self.capacity)

    def add(self, x):
        self.buffer.append(x)

    def load(self):
        states = list()
        for _ in range(self.capacity):
            states.append(self.buffer.popleft())
        return states

    def get_data(self):
        lasers = list()
        polar_goal = list()
        for idx, frame in enumerate(self.buffer):
            lasers.append(frame.lasers)
            polar_goal.append(frame.polar_goal)
        return lasers, polar_goal

    def clear(self):
        self.buffer.clear()

    def display(self):
        axes = []
        width = 5
        height = 5
        rows = self.capacity
        columns = 1
        fig = plt.figure()
        for idx, obs in enumerate(self.buffer):
            lasers = torch.tensor(obs.lasers).unsqueeze(0)
            axes.append(fig.add_subplot(rows, columns, idx + 1))
            subplot_title = ("Frame " + str(idx))
            axes[-1].set_title(subplot_title)
            plt.imshow(lasers.permute(0,1), cmap = 'gray')
        fig.tight_layout()
        plt.show()

    def print_lasers(self):
        for idx, frame in enumerate(self.buffer):
            print 'Frame(',idx + 1,'):',frame.lasers

    def print_polar_goal(self):
        for idx, frame in enumerate(self.buffer):
            print 'Frame(',idx + 1,'):',frame.polar_goal

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return str(self.buffer)

class Observation(object):
    def __init__(self, lasers, polar_goal):
        self._lasers = lasers
        self._polar_goal = polar_goal

    def __repr__(self):
        return 'lasers: '+str(self._lasers)+'\npolar_goal: '+str(self._polar_goal)+'\n'

    def __eq__(self, other):
        return self._lasers == other.lasers and self._polar_goal == other.polar_goal

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def polar_goal(self):
        return self._polar_goal

    @property
    def lasers(self):
        return self._lasers