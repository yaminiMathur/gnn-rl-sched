### Import Libraries
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple

class Agent():


    def load(self, file:str, exploration_rate=False, new_rate=0.6):
        path = self.save_dir+"/"+file
        loaded_file = torch.load(path)
        self.net.load_state_dict(loaded_file["model"])
        if exploration_rate:
            self.exploration_rate = loaded_file["exploration_rate"]
        else:
            self.exploration_rate = new_rate
        self.net.eval()

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

### Cache Replay Buffer
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


    ### Main Training Loop
    def learn_batch(selfenv, buffer=None, behavior=None, learning_history=[]):
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


class GCN(nn.Module):

    def __init__(self, aggregator, features=5, hidden_layer_size=5, embedding_size=1):
        super(GCN, self).__init__()
        
        # Simple Graph Conv 
        # self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        # self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size)
        # self.conv3 = GraphConv(in_feats=hidden_layer_size, out_feats=embedding_size)

        # Using Graph SAGE to get embedding using the neighbours
        self.conv1 = SAGEConv(in_feats=features, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv2 = SAGEConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv3 = SAGEConv(in_feats=hidden_layer_size, out_feats=embedding_size, aggregator_type=aggregator)

    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.sigmoid(h)
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        h = self.conv3(g, h)
        h = torch.sigmoid(h)

        return h

