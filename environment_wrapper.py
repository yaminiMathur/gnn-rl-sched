from operator import le
from spark_env.env import *
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import time

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
    
    def __init__(self, view_range=50, reset_prob=5e-7, max_exec=100, env_len = 50, turn = 40) -> None:

        # Set pre built environment
        self.env = Environment()
        
        # environment parameters
        self.reset_prob = reset_prob
        self.max_exec = max_exec

        # wrapper parameters
        self.range = view_range
        self.env_len = env_len
        self.offset = 0
        self.turn = turn

        self.frontier_nodes = []
        self.source_exec = max_exec
        self.logits = None
        self.gnn = GCN().to(cuda)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=1e-3)
        
        # create prebuilt environment
        self.reset()
    
    # reset the environment to a new seed
    def reset(self, min_time=1, max_time=2):
        seed = np.random.randint(min_time, max_time)
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

        # calculate the logits and filter the required indices
        # based on the number of nodes the agent can see
        logits = self.gnn(G.to(cuda), node_inputs.to(cuda))
            
        required_indices = []
        for i in range(1, self.range+1):
            if len(required_indices) == len(frontier_nodes):
                break
            index = (self.offset+i)%len(frontier_nodes)
            required_indices.append(leaf_nodes[index])

        # pad the output with 0 if 50 nodes are not available
        # and add the number of source executors remaining to the vector
        logits  = logits[required_indices]
        padding = torch.zeros(self.range-len(logits), 1).to(cuda)
        logits  = torch.cat([logits, padding])
        logits  = logits.flatten()

        return logits

    # perform an action and return the resultant state and reward
    def step(self, action, early_stop=True):

        # get the direction, job and limit
        node, limit = action
        direction = 1
        job = node % 50

        # count the frontier nodes and 
        frontier_node_count = len(self.frontier_nodes)

        # if no more frontier nodes then clear the executors
        if frontier_node_count == 0:
            reward, done = self.env.step(None, self.max_exec)
            state = self.observe()
            return state, reward, early_stop or done

        # if index is greater than the number of jobs, limit it to the length of frontier nodes
        index = (self.offset + job) % self.range
        if index >= frontier_node_count:
            index = frontier_node_count-1

        if limit > self.source_exec :
            limit = self.source_exec
        limit = max(1, limit)

        # update the view offset to check for more jobs i.e stay or move right
        self.offset += direction*self.turn
        self.offset = self.offset % self.range

        # take a step and observe the reward, completion and the state from the old environement
        reward, done = self.env.step(self.frontier_nodes[index], limit)
        state = self.observe()
        
        # return None, None, None
        return state, reward, done

class GraphWrapper:
    
    def __init__(self, view_range=50, reset_prob=5e-7, max_exec=100, env_len = 50, turn = 40) -> None:

        # Set pre built environment
        self.env = Environment()
        
        # environment parameters
        self.reset_prob = reset_prob
        self.max_exec = max_exec

        # wrapper parameters
        self.range = view_range
        self.env_len = env_len
        self.offset = 0
        self.turn = turn

        self.frontier_nodes = []
        self.leaf_nodes = []
        self.source_exec = max_exec
        self.logits = None
        
        # create prebuilt environment
        self.reset()
    
    # reset the environment to a new seed
    def reset(self, min_time=1, max_time=2):
        seed = np.random.randint(min_time, max_time)
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

        # calculate the logits and filter the required indices
        # based on the number of nodes the agent can see
        # logits = self.gnn(G.to(cuda), node_inputs.to(cuda))

        return G, node_inputs, leaf_nodes #logits

    # perform an action and return the resultant state and reward
    def step(self, action, early_stop=True):

        # get the direction, job and limit
        index, limit = action

        # count the frontier nodes and 
        frontier_node_count = len(self.frontier_nodes)

        # if no more frontier nodes then clear the executors
        if frontier_node_count == 0:
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
        
        # return None, None, None
        return state, reward, done

    def get_networkx(self):
        return self.env.G, self.env.pos