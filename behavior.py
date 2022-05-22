
### Import Libraries
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, datetime
from matplotlib import pyplot as plt
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from numpy.random import randint
import dgl
from param import args
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple, deque

from udrl import GCN, ReplayBuffer, MetricLogger
import agent



### Behavior Function
class Behavior(nn.Module):
    '''
    Behavour function that produces actions based on a state and command.
    NOTE: At the moment I'm fixing the amount of units and layers.
    TODO: Make hidden layers configurable.
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        command_scale (List of float)
    '''
    
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 command_scale = [1, 1]):
        super().__init__()
        
        self.command_scale = torch.FloatTensor(command_scale).to(device)
        
        self.state_fc = nn.Sequential(nn.Linear(state_size, 64), 
                                      nn.Tanh())
        
        self.command_fc = nn.Sequential(nn.Linear(2, 64), 
                                        nn.Sigmoid())
        
        self.output_fc = nn.Sequential(nn.Linear(64, 128), 
                                       nn.ReLU(), 
#                                        nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
#                                        nn.Dropout(0.2),
                                       nn.Linear(128, 128), 
                                       nn.ReLU(), 
                                       nn.Linear(128, action_size))
        
        self.to(device)
        
    
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.command_scale)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)
    
    def action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- stochastic action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()

    ### On Assist
    def greedy_action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- greedy action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())
    
    def init_optimizer(self, optim=Adam, lr=0.003):
        '''Initialize GD optimizer
        
        Params:
            optim (Optimizer) -- default Adam
            lr (float) -- default 0.003
        '''
        
        self.optim = optim(self.parameters(), lr=lr)