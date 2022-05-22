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
from behavior import Behavior



class Agent():
    ### Hyperparameters ###

    # Number of iterations in the main loop
    n_main_iter = 700

    # Number of (input, target) pairs per batch used for training the behavior function
    batch_size = 768

    # Scaling factor for desired horizon input
    horizon_scale = 0.01

    # Number of episodes from the end of the replay buffer used for sampling exploratory
    # commands
    last_few = 75

    # Learning rate for the ADAM optimizer
    learning_rate = 0.0003

    # Number of exploratory episodes generated per step of UDRL training
    n_episodes_per_iter = 20

    # Number of gradient-based updates of the behavior function per step of UDRL training
    n_updates_per_iter = 100

    # Number of warm up episodes at the beginning of training
    n_warm_up_episodes = 10

    # Maximum size of the replay buffer (in episodes)
    replay_size = 500

    # Scaling factor for desired return input
    return_scale = 0.02

    # Evaluate the agent after `evaluate_every` iterations
    evaluate_every = 10

    # Target return before breaking out of the training loop
    target_return = 200

    # Maximun reward given by the environment
    max_reward = 250

    # Maximun steps allowed
    max_steps = 300

    # Reward after reaching `max_steps` (punishment, hence negative reward)
    max_steps_reward = -50

    # Hidden units
    hidden_size = 32

    # Times we evaluate the agent
    n_evals = 1

    # Will stop the training when the agent gets `target_return` `n_evals` times
    stop_on_solved = False

    
    #######################

    def __init__(self, save_dir="./models", assist=True, assist_p=(1, 7), aggregator="mean"):
        print("Initializing Agent... ")
            # Helper function to create episodes as namedtuple
        make_episode = namedtuple('Episode', 
                                field_names=['states', 
                                            'actions', 
                                            'rewards', 
                                            'init_command', 
                                            'total_return', 
                                            'length', 
                                            ])
        self.save_dir = save_dir

        # DNN to predict the most optimal action
        #self.net = Net(aggregator).float()
        #self.net = self.net.to(cuda)
        self.aggregator = aggregator

        self.exploration_rate = 1
        self.exploration_rate_decay = args.exploration_rate_decay # 0.9999975 # 0.999992 
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=100000)
        self.batch_size = args.batch_size

        self.save_every = 1e3  # no. of experiences

        self.gamma = args.gamma

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        #self.loss_fn = torch.nn.SmoothL1Loss()

        #self.burnin = args.burnin            # min. experiences before training
        self.learn_every = args.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = args.sync_every    # no. of experiences between Q_target & Q_online sync

        self.assist = assist
        self.assist_range = assist_p
        self.softmax = nn.Softmax()

    def load(self, file:str, new_rate=0.6):
        path = self.save_dir+"/"+file
        loaded_file = torch.load(path)
        self.net.load_state_dict(loaded_file["model"])
        # if exploration_rate:
        #     self.exploration_rate = loaded_file["exploration_rate"]
        # else:
        #     self.exploration_rate = new_rate
        self.net.eval()
    
    ### Sample Exploratory Commands
    def sample_command(buffer, last_few):
        '''Sample a exploratory command
        
        Params:
            buffer (ReplayBuffer)
            last_few:
                how many episodes we're gonna look at to calculate 
                the desired return and horizon.
        
        Returns:
            List of float -- command
        '''
        if len(buffer) == 0: return [1, 1]
        
        # 1.
        commands = buffer.get(last_few)
        
        # 2.
        lengths = [command.length for command in commands]
        desired_horizon = round(np.mean(lengths))
        
        # 3.
        returns = [command.total_return for command in commands]
        mean_return, std_return = np.mean(returns), np.std(returns)
        desired_return = np.random.uniform(mean_return, mean_return+std_return)
        
        return [desired_return, desired_horizon]

    ### Cache Replay Buffer
    def initialize_replay_buffer(self, env, replay_size, n_episodes, last_few):
        '''
        Initialize replay buffer with warm-up episodes using random actions.
        See section 2.3.1
        
        Params:
            replay_size (int)
            n_episodes (int)
            last_few (int)
        
        Returns:
            ReplayBuffer instance
            
        '''
        
        # This policy will generate random actions. Won't need state nor command
        random_policy = lambda state, command: np.random.randint(env.action_space.n)
        
        buffer = ReplayBuffer(replay_size)      
        
        for i in range(n_episodes):
            command = self.sample_command(buffer, last_few)
            episode = self.generate_episode(env, random_policy, command) # See Algorithm 2
            buffer.add(episode)
        
        buffer.sort()
        print("\n\nInitialized Replay Buffer...")
        return buffer
