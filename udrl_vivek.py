### Import Libraries
from typing import List, Tuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from numpy import array, load
import random
from environment_wrapper import *
from dgl.nn.pytorch import SAGEConv
import dgl
import numpy as np

warnings.filterwarnings("ignore")

# actions
RANDOM = 1
BEHAVIOUR_ACT = 2
BEHAVIOUR_GREEDY = 3

# config -max scheduling decisions
MAX_STEPS = 2500
MAX_STEPS_REWARD = -50
MAX_REWARD = 0

# ------------------------------------------------------------------------------------------------

### GCN function to get the embedding
class GCN(nn.Module):

    def __init__(self, aggregator, features, hidden_layer_size, embedding_size, device="cpu"):
        super(GCN, self).__init__()
        
        # Simple Graph Conv 
        # self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        # self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=hidden_layer_size)

        # Using Graph SAGE to get embedding using the neighbours
        self.conv1 = SAGEConv(in_feats=features, out_feats=hidden_layer_size, aggregator_type=aggregator)
        self.conv2 = SAGEConv(in_feats=hidden_layer_size, out_feats=embedding_size, aggregator_type=aggregator)
        self.to(device)
        
    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.sigmoid(h)
        h = self.conv2(g, h)
        return torch.sigmoid(h)

# -------------------------------------------------------------------------------------------------

### Behavior Function
class Behaviour(nn.Module):
    
    def __init__(self, aggregator='pool', features=5, hidden_layer_size=5, 
                embedding_size=10, command_scale = [1, 1], device='cpu', prob=0):

        super().__init__()

        # Check which device to be used for training and test
        self.device = device
        self.prob = prob
        
        self.command_scale = torch.FloatTensor(command_scale).to(self.device)
        
        # Using Graph SAGE to get embedding for the graph
        self.gcn = GCN(aggregator, features, hidden_layer_size, embedding_size, device)
        # Use a sequential layer to get ecoding of the reward and command
        self.command_fn = nn.Sequential(nn.Linear(2, embedding_size)).to(device)

        # Output function
        self.output_nn = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Sigmoid(),

            nn.Linear(32, 32),
            nn.Sigmoid(),

            nn.Linear(32, 32),
            nn.Sigmoid(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)
    
    #  Neural Network -
    #
    #  [10]- node embedding [10] - graph summary  |  [20] - command
    #  [20] - graph input  |  [20] - command
    #  [20] - graph input * command ----- neural network input 
    #  [input - 20] -- layer 1 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 2 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 3 (sigmoid) -- [output - 32]
    #  [input - 32] -- layer 4 (sigmoid) -- [output - 1] -- Probability of scheduling node

    # forward propagation for the neural network
    def forward(self, G, node_inputs, leaves, command):
        # Get the gnn embedding of the graph
        gnn_embeddings = self.gcn(dgl.batch(G).to(self.device), node_inputs.to(self.device))
        # compute the encoding of the commnd
        encoded_command = self.command_fn(torch.tensor(command).to(self.device))
        leaf_embeddings = []

        # filter out the leaf nodes
        prev = 0
        current = 0
        for g, l in zip(G, leaves):
            current += g.number_of_nodes()
            # seperate current graph
            g_embeddings = gnn_embeddings[prev:current]
            # filter out only the leaves and append to list
            leaf_embeddings.append(g_embeddings[l])
            # change the new prev
            prev = current

        # assign the graph embeddings with the current leaf embeddings
        gnn_embeddings = torch.cat(leaf_embeddings)
        # compute the product with command as per upside down RL and pass 
        # it through the output function
        product = torch.mul(gnn_embeddings, encoded_command)
        return self.output_nn(product)
    
    # Compute the action from a state and command
    # Note: this is only for computing a single action and not for training
    def action(self, state, command):
        G, node_inputs, leaf_nodes = state
        shape = node_inputs.shape
        assert len(shape) < 2 or shape[1] == 1
        # add the encoded command to the gnn embeddings
        embeddings = self.forward(G, node_inputs, leaf_nodes, command)
        embeddings = embeddings.flatten()
        return embeddings.argmax().item()

    # Compute the best action based on the highest current probabilities
    def greedy_action(self, state, command):
        G, node_inputs, leaf_nodes = state
        gnn_embeddings = self.get_gnn_embedding(G.to(self.device), node_inputs.to(self.device))
        logits = self.forward(gnn_embeddings[leaf_nodes], command)
        probs = F.softmax(logits)
        return torch.argmax(probs).item()

    # Get the random action for state
    def random_action(self, env:GraphWrapper):
        G, node_inputs, leaf_nodes = env.observe()
        action = leaf_nodes[(random.randint(0, len(leaf_nodes)-1))]
        if random.randint(0, 100-1) < self.prob :
            action = env.auto_step()
        return action

    # save the model parameters
    def save(self, filename):
        torch.save(self.state_dict, "behaviour_"+filename)
    
    # load model parameters
    def load(self, filename):
        self.state_dict(torch.load("behaviour_"+filename))

# -----------------------------------------------------------------------------------------------------

class Episode :

    def __init__(self, init_command:List) :
        self.list = []
        self.total_reward = 0   
        self.time_steps = 0
        self.init_command = init_command

    def add_iteration(self, step:Tuple):
        # G, node_inputs, leaf_nodes, action, reward
        if len(step) != 5:
            print("Length of each episode has to be 5")
            return
        self.total_reward += step[4] 
        self.time_steps += 1
        self.list.append(step)

    def create_batch(self, observations):
        graph_list = []; node_input_list = []
        leaf_node_list = []; action_list = []; reward_sum = 0
        for observation in observations:
            G, node_inputs, leaf_nodes, action, reward = observation
            graph_list.append(G)
            node_input_list.append(node_inputs)
            leaf_node_list.append(leaf_nodes)
            action_list.append(torch.FloatTensor([int(action == leaf) for leaf in leaf_nodes]))
            reward_sum += reward
        action_list = torch.cat(action_list)
        action_shape = action_list.shape
        return graph_list, torch.cat(node_input_list), torch.reshape(action_list, (action_shape[0], 1)), leaf_node_list, reward_sum

    def sample(self):
        total_len = len(self.list)
        start = random.randint(0, total_len-2)
        end = min(random.randint(start+1, start+25), total_len-1)
        horizon = end-start
        graphs, inputs, actions, leaves, tot_reward = self.create_batch(self.list[start:end])
        return graphs, inputs, actions, leaves, [tot_reward, horizon]

    def __lt__(self, other):
        return self.total_reward > other.total_reward

    def __len__(self):
        return len(self.list)

### Replay Buffer for episodes
class Memory:

    def __init__(self, size=0):
        self.size = size
        self.buffer = []
        
    def add_episode(self, episode:Episode):
        self.buffer.append(episode)
    
    def get(self, num):
        return self.buffer[-num:]
    
    def sample(self, batch_size):
        return np.random.choice(self.buffer, size=batch_size)
    
    def sort(self):
        self.buffer.sort()
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------------------------------------------

class Trainer : 

    def __init__(self, device, optimizer=Adam, lr=0.003, mem_size=10) -> None:
        self.memory = Memory()
        self.behaviour = Behaviour(device=device)
        self.env = GraphWrapper()
        self.optimizer = optimizer(self.behaviour.parameters(), lr=lr)
        self.device = device
        print("Generating training data")
        for i in range(mem_size):
            self.memory.add_episode(self.run_episode())
        print("completed generating training data")

    def run_episode(self, policy=RANDOM, init_command=[0, 1000], seed=None, max_steps=100):
        episode = Episode(init_command=init_command)
        command = init_command.copy()
        if not seed :
            seed = random.randint(10000, 100000)
        
        print("Seed : ", seed)
        self.env.reset(seed)
        state = self.env.observe()
        done = False
        step = 0

        print("Started episode")
        while not done and  step < max_steps:

            index = None
            action = None
            if policy == RANDOM :
                index = self.behaviour.random_action(self.env)
                action = (index, 2)
            elif policy == BEHAVIOUR_ACT :
                index = self.behaviour.action(state, command)
                action =  (index, 1)
            else :
                index = self.behaviour.greedy_action(state, command)
                action = (index, 1)

            next_state, reward, done = self.env.step(action)
            if episode.time_steps > MAX_STEPS :
                done = True
                reward = MAX_STEPS_REWARD
            
            G, node_inputs, leaf_nodes = state  
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])

            state = next_state
            command[0] = min(command[0]-reward, MAX_REWARD) # desired reward
            command[1] = max(command[1]-1, 1)               # desired time frame
            step += 1
        
        print("Ended episode")
        return episode

    def train(self, steps:int, batch_size=1):
        loss_list = []
        print("started training batches")
        for step in range(steps):
            print("Batch :", step)
            episodes = self.memory.sample(batch_size)
            for episode in episodes:
                print("Episode started")
                graphs, inputs, actions, leaves, command = episode.sample()
                new_predictions = self.behaviour(graphs, inputs, leaves, command)
                loss = F.cross_entropy(new_predictions, actions.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss.item())
                raise Exception
                loss_list.append(loss.item())
                print("Episode ended")
            print("Batch Complete", step)
        print("training batch complete")

trainer = Trainer(device='cuda', mem_size=1)
trainer.train(steps=1)