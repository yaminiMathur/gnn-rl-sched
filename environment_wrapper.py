from operator import le
from spark_env.env import Environment
from param import args
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import time
import torch
import numpy as np

cuda = args.cuda

class GCN(nn.Module):

    def __init__(self, features=5, hidden_layer_size=5, embedding_size=1):
        super(GCN, self).__init__()
        
        self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=embedding_size)
        # self.conv3 = GraphConv(hidden_layer_size, embedding_size)

    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        return h

class EnvironmentWrapper:
    
    def __init__(self, view_range=50, reset_prob=5e-7, env_len = 50) -> None:

        # Set pre built environment
        self.env = Environment()
        
        # environment parameters
        self.reset_prob = reset_prob
        self.max_exec = args.exec_cap

        # wrapper parameters
        self.range = view_range
        self.env_len = env_len

        self.frontier_nodes = []
        self.leaf_nodes = []
        self.source_exec = args.exec_cap
        self.gnn = GCN().to(cuda)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=1e-3)
        
        # create prebuilt environment
        self.reset()
    
    # reset the environment to a new seed
    def reset(self, minseed=1, maxseed=1234):
        seed = np.random.randint(minseed, maxseed)
        self.env.seed(seed)
        self.env.reset(max_time=np.random.geometric(self.reset_prob))
        self.observe()

    # observe and decode an observation into usable paramaters for the agent
    # this involves embedding the graph nodes into 10 dim vectors
    def observe(self):
        # get the new observation from the environement
        G, frontier_nodes, leaf_nodes, num_source_exec, node_inputs = self.env.observe()
        mask = torch.zeros(G.number_of_nodes()).to(cuda)
        mask[leaf_nodes] = 1
        mask = mask.bool()

        # reset the frontier nodes and the number of free executors
        self.frontier_nodes = frontier_nodes
        self.source_exec = num_source_exec
        self.leaf_nodes = []

        # calculate the logits and filter the required indices
        # based on the number of nodes the agent can see
        logits = self.gnn(G.to(cuda), node_inputs.to(cuda))         
        logits = logits[mask]
        padding_required = max(self.range - len(logits), 0)
        padding = torch.zeros(padding_required, 1).to(cuda)
        logits  = torch.cat([logits, padding])
        logits  = logits.flatten()

        return logits.detach()

    # perform an action and return the resultant state and reward
    def step(self, action, early_stop=True):

        # get the direction, job and limit
        node, limit = action
        node = self.leaf_nodes[node]

        # count the frontier nodes and 
        frontier_node_count = len(self.frontier_nodes)

        # if no more frontier nodes then clear the executors
        if frontier_node_count == 0:
            reward, done = self.env.step(None, self.max_exec)
            state = self.observe()
            return state, reward, early_stop or done

        # take a step and observe the reward, completion and the state from the old environement
        reward, done = self.env.step(self.frontier_nodes[node], limit)
        state = self.observe()
        
        # return None, None, None
        return state, reward, done


class GraphWrapper:
    
    def __init__(self, reset_prob=5e-7) -> None:

        # Set pre built environment
        self.env = Environment()
        
        # environment parameters
        self.reset_prob = reset_prob
        self.max_exec = args.exec_cap

        self.frontier_nodes = []
        self.leaf_nodes = []
        self.source_exec = args.exec_cap
    
    # reset the environment to a new seed
    def reset(self, seed:int):
        self.env.seed(seed)
        self.env.reset(max_time=np.random.geometric(self.reset_prob))
        self.offset = 0
        self.logits = False
        self.observe()

    # observe and decode an observation into usable paramaters for the agent
    # this involves embedding the graph nodes into 10 dim vectors
    def observe(self):
        # get the new observation from the environement
        G, frontier_nodes, leaf_nodes, num_source_exec, node_inputs = self.env.observe()

        # reset the frontier nodes and the number of free executors
        self.frontier_nodes = frontier_nodes
        self.source_exec = num_source_exec
        self.leaf_nodes = leaf_nodes

        return G, node_inputs, leaf_nodes

    # perform an action and return the resultant state and reward
    def step(self, action, early_stop=True):

        # set the job index as per the leaf nodes
        index, limit = action

        # count the frontier nodes and 
        frontier_node_count = len(self.frontier_nodes)

        # if no more frontier nodes then clear the executors
        if frontier_node_count == 0 or index < 0:
            reward, done = self.env.step(None, self.max_exec)
            state = self.observe()
            return state, reward, early_stop or done
            
        # limits for the number of executors
        if limit > self.source_exec :
            limit = self.source_exec
        limit = max(1, limit)

        # take a step and observe the reward, completion and the state from the old environement
        reward, done = self.env.step(self.frontier_nodes[index], limit)
        state = self.observe()
        
        return state, reward, done
    
    # to get and siplay the graph
    def get_networkx(self):
        return self.env.G, self.env.pos