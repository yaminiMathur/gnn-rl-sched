### Import Libraries
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
from numpy import load
from environment_wrapper import *
from numpy.random import randint
import time
from dgl.nn.pytorch import SAGEConv

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

warnings.filterwarnings("ignore")

# Traing on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = GraphWrapper()

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n
# print('State size: {}'.format(state_size))
# print('Action size: {}'.format(action_size))

# Helper function to create episodes as namedtuple
make_episode = namedtuple('Episode', 
                          field_names=['states', 
                                       'actions', 
                                       'rewards', 
                                       'init_command', 
                                       'total_return', 
                                       'length', 
                                       ])
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

### Generate Episode

def generate_episode(env, policy, init_command=[1, 1]):
    '''
    Generate an episode using the Behaviour function.
    
    Params:
        env (OpenAI Gym Environment)
        policy (func)
        init_command (List of float) -- default [1, 1]
    
    Returns:
        Namedtuple (states, actions, rewards, init_command, total_return, length)
    '''
    
    command = init_command.copy()
    desired_return = command[0]
    desired_horizon = command[1]
    
    states = []
    actions = []
    rewards = []
    
    time_steps = 0
    done = False
    total_rewards = 0
    state = env.reset().tolist()
    
    while not done:
        state_input = torch.FloatTensor(state).to(device)
        command_input = torch.FloatTensor(command).to(device)
        action = policy(state_input, command_input)
        next_state, reward, done, _ = env.step(action)
        
        # Modifying a bit the reward function punishing the agent, -100, 
        # if it reaches hyperparam max_steps. The reason I'm doing this 
        # is because I noticed that the agent tends to gather points by 
        # landing the spaceshipt and getting out and back in the landing 
        # area over and over again, never switching off the engines. 
        # The longer it does that the more reward it gathers. Later on in 
        # the training it realizes that it can get more points by turning 
        # off the engines, but takes more epochs to get to that conclusion.
        if not done and time_steps > max_steps:
            done = True
            reward = max_steps_reward
        
        # Sparse rewards. Cumulative reward is delayed until the end of each episode
#         total_rewards += reward
#         reward = total_rewards if done else 0.0
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state.tolist()
        
        # Clipped such that it's upper-bounded by the maximum return achievable in the env
        desired_return = min(desired_return - reward, max_reward)
        
        # Make sure it's always a valid horizon
        desired_horizon = max(desired_horizon - 1, 1)
    
        command = [desired_return, desired_horizon]
        time_steps += 1
        
    return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

def generate_episodes(env, behavior, buffer, n_episodes, last_few):
    '''
    1. Sample exploratory commands based on replay buffer
    2. Generate episodes using Algorithm 2 and add to replay buffer
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_episodes (int)
        last_few (int):
            how many episodes we use to calculate the desired return and horizon
    '''
    
    stochastic_policy = lambda state, command: behavior.action(state, command)
    
    for i in range(n_episodes_per_iter):
        command = sample_command(buffer, last_few)
        episode = generate_episode(env, stochastic_policy, command) # See Algorithm 2
        buffer.add(episode)
    
    # Let's keep this buffer sorted
    buffer.sort()

### Utility Functions
def initialize_replay_buffer(replay_size, n_episodes, last_few):
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
        command = sample_command(buffer, last_few)
        episode = generate_episode(env, random_policy, command) # See Algorithm 2
        buffer.add(episode)
    
    buffer.sort()
    return buffer

def initialize_behavior_function(state_size, 
                                 action_size, 
                                 hidden_size, 
                                 learning_rate, 
                                 command_scale):
    '''
    Initialize the behaviour function. See section 2.3.2
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        learning_rate (float)
        command_scale (List of float)
    
    Returns:
        Behavior instance
    
    '''
    
    behavior = Behavior(state_size, 
                        action_size, 
                        hidden_size, 
                        command_scale)
    
    behavior.init_optimizer(lr=learning_rate)
    
    return behavior



### Replay Buffer

class ReplayBuffer():
    '''
    Replay buffer containing a fixed maximun number of trajectories with 
    the highest returns seen so far
    
    Params:
        size (int)
    
    Attrs:
        size (int)
        buffer (List of episodes)
    '''
    
    def __init__(self, size=0):
        self.size = size
        self.buffer = []
        
    def add(self, episode):
        '''
        Params:
            episode (namedtuple):
                (states, actions, rewards, init_command, total_return, length)
        '''
        
        self.buffer.append(episode)
    
    def get(self, num):
        '''
        Params:
            num (int):
                get the last `num` episodes from the buffer
        '''
        
        return self.buffer[-num:]
    
    def random_batch(self, batch_size):
        '''
        Params:
            batch_size (int)
        
        Returns:
            Random batch of episodes from the buffer
        '''
        
        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]
    
    def sort(self):
        '''Keep the buffer sorted in ascending order by total return'''
        
        key_sort = lambda episode: episode.total_return
        self.buffer = sorted(self.buffer, key=key_sort)[-self.size:]
    
    def save(self, filename):
        '''Save the buffer in numpy format
        
        Param:
            filename (str)
        '''
        
        np.save(filename, self.buffer)
    
    def load(self, filename):
        '''Load a numpy format file
        
        Params:
            filename (str)
        '''
        
        raw_buffer = np.load(filename)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]
    
    def __len__(self):
        '''
        Returns:
            Size of the buffer
        '''
        return len(self.buffer)


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
                 aggregator, features=5, hidden_layer_size=5, embedding_size=10,
                 command_scale = [1, 1]):

        ### TODO: Replace by Graph Neural Network
        super().__init__()
        
        self.command_scale = torch.FloatTensor(command_scale).to(device)
               # Using Graph SAGE to get embedding using the neighbours
        self.conv1 = SAGEConv(in_feats=features, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv2 = SAGEConv(in_feats=hidden_layer_size, out_feats=embedding_size, aggregator_type=aggregator)
        self.command_fn = nn.Sequential(nn.Linear(2, 10))
        
        self.to(device)
        
    
    def forward(self, g, inputs, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        h = inputs
        h = self.conv1(g, h)
        h = torch.sigmoid(h)
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        h = self.conv3(g, h)
        h = torch.sigmoid(h)

        return h
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
    
    def save(self, filename):
        '''Save the model's parameters
        Param:
            filename (str)
        '''
        
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        '''Load the model's parameters
        
        Params:
            filename (str)
        '''
        
        self.load_state_dict(torch.load(filename))


def UDRL(env, buffer=None, behavior=None, learning_history=[]):
    '''
    Upside-Down Reinforcement Learning main algrithm
    
    Params:
        env (OpenAI Gym Environment)
        buffer (ReplayBuffer):
            if not passed in, new buffer is created
        behavior (Behavior):
            if not passed in, new behavior is created
        learning_history (List of dict) -- default []
    '''
    
    if buffer is None:
        buffer = initialize_replay_buffer(replay_size, 
                                          n_warm_up_episodes, 
                                          last_few)
    
    if behavior is None:
        behavior = initialize_behavior_function(state_size, 
                                                action_size, 
                                                hidden_size, 
                                                learning_rate, 
                                                [return_scale, horizon_scale])
    
    for i in range(1, n_main_iter+1):
        mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
        
        print('Iter: {}, Loss: {:.4f}'.format(i, mean_loss), end='\r')
        
        # Sample exploratory commands and generate episodes
        generate_episodes(env, 
                          behavior, 
                          buffer, 
                          n_episodes_per_iter,
                          last_few)
        
        if i % evaluate_every == 0:
            command = sample_command(buffer, last_few)
            mean_return = evaluate_agent(env, behavior, command)
            
            learning_history.append({
                'training_loss': mean_loss,
                'desired_return': command[0],
                'desired_horizon': command[1],
                'actual_return': mean_return,
            })
            
            if stop_on_solved and mean_return >= target_return: 
                break
    
    return behavior, buffer, learning_history

def train_behavior(behavior, buffer, n_updates, batch_size):
    '''Training loop
    
    Params:
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_updates (int):
            how many updates we're gonna perform
        batch_size (int):
            size of the bacth we're gonna use to train on
    
    Returns:
        float -- mean loss after all the updates
    '''
    all_loss = []
    for update in range(n_updates):
        episodes = buffer.random_batch(batch_size)
        
        batch_states = []
        batch_commands = []
        batch_actions = []
        
        for episode in episodes:
            T = episode.length
            t1 = np.random.randint(0, T)
            t2 = np.random.randint(t1+1, T+1)
            dr = sum(episode.rewards[t1:t2])
            dh = t2 - t1
            
            st1 = episode.states[t1]
            at1 = episode.actions[t1]
            
            batch_states.append(st1)
            batch_actions.append(at1)
            batch_commands.append([dr, dh])
        
        batch_states = torch.FloatTensor(batch_states).to(device)
        batch_commands = torch.FloatTensor(batch_commands).to(device)
        batch_actions = torch.LongTensor(batch_actions).to(device)
        
        pred = behavior(batch_states, batch_commands)
        
        loss = F.cross_entropy(pred, batch_actions)
        
        behavior.optim.zero_grad()
        loss.backward()
        behavior.optim.step()
        
        all_loss.append(loss.item())
    
    return np.mean(all_loss)

def evaluate_agent(env, behavior, command, render=False):
    '''
    Evaluate the agent performance by running an episode
    following Algorithm 2 steps
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        command (List of float)
        render (bool) -- default False:
            will render the environment to visualize the agent performance
    '''
    behavior.eval()
    
    print('\nEvaluation.', end=' ')
        
    desired_return = command[0]
    desired_horizon = command[1]
    
    print('Desired return: {:.2f}, Desired horizon: {:.2f}.'.format(desired_return, desired_horizon), end=' ')
    
    all_rewards = []
    
    for e in range(n_evals):
        
        done = False
        total_reward = 0
        state = env.reset().tolist()
    
        while not done:
            if render: env.render()
            
            state_input = torch.FloatTensor(state).to(device)
            command_input = torch.FloatTensor(command).to(device)

            action = behavior.greedy_action(state_input, command_input)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state.tolist()

            desired_return = min(desired_return - reward, max_reward)
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
        
        if render: env.close()
        
        all_rewards.append(total_reward)
    
    mean_return = np.mean(all_rewards)
    print('Reward achieved: {:.2f}'.format(mean_return))
    
    behavior.train()
    
    return mean_return


# Number of iterations in the main loop
n_main_iter = 5000

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
return_scale = 20

# Evaluate the agent after `evaluate_every` iterations
evaluate_every = 10

# Target return before breaking out of the training loop
target_return = 1

# Maximun reward given by the environment
max_reward = 1

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