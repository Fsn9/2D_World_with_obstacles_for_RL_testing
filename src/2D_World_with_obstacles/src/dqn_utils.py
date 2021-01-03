# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
# Python
from collections import deque
from random import sample
# ROS
import rospy
# Matplotlib
import matplotlib.pyplot as plt

dtype = torch.float32

class DCNN1D(torch.nn.Module):
    def __init__(self, input_dim, input_channels, output_dim, learning_rate, momentum):
        super(DCNN1D, self).__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_frames = self.input_channels
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
        self.momentum = momentum
        self.criterion = nn.MSELoss() # change to huber loss
        self.optimizer = optimizers.RMSprop(self.parameters(), lr = self.learning_rate, momentum = self.momentum)

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

class DFF(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DFF, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # define layers
    def forward(self, x):
        pass

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

        self._current_lasers = torch.cat(tuple(self.current_lasers_batch_list), dim = 0).float()
        self._current_polar_goals = torch.cat(tuple(self.current_polar_goal_batch_list), dim = 0).float()
        self._next_lasers = torch.cat(tuple(self.next_lasers_batch_list), dim = 0).float()
        self._next_polar_goals = torch.cat(tuple(self.next_polar_goal_batch_list), dim = 0).float()
        self._actions = torch.tensor(self.action_batch_list).view(self.batch_size, 1)
        self._rewards = torch.tensor(self.reward_batch_list, dtype = dtype).view(self.batch_size, 1)
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

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return str(self.buffer)

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
            print('Frame(',idx+1,'):',frame.lasers)

    def print_polar_goal(self):
        for idx, frame in enumerate(self.buffer):
            print('Frame(',idx+1,'):',frame.polar_goal)


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
 