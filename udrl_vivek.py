### Import Libraries
from asyncio.log import logger
from dataclasses import replace
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
import datetime
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# actions
RANDOM = 1
BEHAVIOUR_ACT = 2
BEHAVIOUR_GREEDY = 3

# config -max scheduling decisions
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
        return h
        # return torch.sigmoid(h)

# -------------------------------------------------------------------------------------------------

### Behavior Function
class Behaviour(nn.Module):
    
    def __init__(self, aggregator='pool', features=5, hidden_layer_size=5, 
                embedding_size=10, command_scale = [1, 1], device='cpu'):

        super().__init__()

        # Check which device to be used for training and test
        self.device = device
        
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
        gnn_embeddings = self.gcn(G.to(self.device), node_inputs.to(self.device))
        encoded_command = self.command_fn(torch.FloatTensor(command).to(self.device))
        product = torch.mul(gnn_embeddings, encoded_command)
        product = self.output_nn(product)
        return leaf_nodes[(product[leaf_nodes]).argmax().item()]

    # Compute the best action based on the highest current probabilities
    def greedy_action(self, state, command):
        G, node_inputs, leaf_nodes = state
        gnn_embeddings = self.get_gnn_embedding(G.to(self.device), node_inputs.to(self.device))
        logits = self.forward(gnn_embeddings[leaf_nodes], command)
        probs = F.softmax(logits)
        return torch.argmax(probs).item()

    # Get the random action for state
    def random_action(self, env:GraphWrapper, prob=50):
        G, node_inputs, leaf_nodes = env.observe()
        action = leaf_nodes[(random.randint(0, len(leaf_nodes)-1))]
        if random.randint(0, 100-1) < prob :
            action = env.auto_step()
        return action

    # save the model parameters
    def save(self, filename):
        torch.save(self.state_dict, "./models/behaviour_"+filename)
    
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
        end = min(random.randint(start+1, start+201), total_len-1)
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
        if len(self.buffer) > self.size:
            self.buffer = np.random.choice(self.buffer, self.size, replace=False)
        self.sort()
    
    def sample(self, batch_size):
        return np.random.choice(self.buffer, 
            size=min(batch_size, len(self.buffer)-1), replace=False)

    def get_possible_command(self, count=10):
        rewards = []
        steps = []
        for episode in self.buffer[-(min(count, self.size)):]:
            rewards.append(episode.total_reward)
            steps.append(episode.time_steps)
        
        possible_horizon = np.mean(steps)
        rewards_std = min(np.std(rewards), 1)
        rewards_mean = np.mean(rewards)
        possible_reward = np.random.uniform(rewards_mean, rewards_mean+rewards_std)

        return [possible_reward, possible_horizon]
         
    def sort(self):
        self.buffer.sort()
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------------------------------------------

class MetricLogger:

    def __init__(self, save_dir="./results", mode="train", version="0", aggregator="mean"):
        save_dir = save_dir+"/"+mode
        self.save_log = save_dir + "/episodes_"+aggregator+"_"+version+".log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot_"+aggregator+"_"+version+".png"
        self.ep_lengths_plot = save_dir + "/length_plot_"+aggregator+"_"+version+".png"
        self.ep_avg_losses_plot = save_dir + "/loss_plot_"+aggregator+"_"+version+".png"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 6)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 6)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 6)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode :  {episode} \n"
            f"Mean Reward :  {mean_ep_reward} \n"
            f"Mean Length :  {mean_ep_length} \n"
            f"Mean Loss :  {mean_ep_loss} \n"
            f"Time Delta :  {time_since_last_record} "
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{mean_ep_reward:10.3f}{mean_ep_length:10.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

# ---------------------------------------------------------------------------------------------------------

class Trainer : 

    def __init__(self, device, optimizer=Adam, lr=0.003, mem_size=50, aggregator="pool") -> None:
        self.memory = Memory(size=mem_size)
        self.behaviour = Behaviour(device=device, aggregator=aggregator)
        self.env = GraphWrapper()
        self.optimizer = optimizer(self.behaviour.parameters(), lr=lr)
        self.device = device
        self.logger = MetricLogger(aggregator=aggregator)
        self.mem_size = mem_size
        self.current_iteration = 0
        print("Generating training data")
        for i in range(round(mem_size/2)):
            self.memory.add_episode(self.run_episode(max_steps=3000))
            print("Added episode to memory :", i)
        print("completed generating training data")

    def run_episode(self, policy=RANDOM, init_command=[1, 1], seed=None, max_steps=1000, assist=50):
        episode = Episode(init_command=init_command)
        command = init_command.copy()
        if not seed :
            seed = random.randint(1, 100000)
        
        print("Seed : ", seed)
        self.env.reset(seed)
        state = self.env.observe()
        done = False
        steps = 0
        loss = -1
        total_reward = 0

        if policy != RANDOM:
            command = self.memory.get_possible_command(count=round(self.mem_size/2))
            loss = self.train_batch(10, 10)
            self.behaviour.save("scheduling_"+str(self.current_iteration)+".pt")

        print("Started episode")
        
        self.logger.init_episode()
        while not done and  steps < max_steps:

            index = None
            action = None
            if policy == RANDOM :
                index = self.behaviour.random_action(self.env, prob=assist)
                action = (index, 1)
            else :
                index = self.behaviour.action(state, command)
                action =  (index, 1)

            next_state, reward, done = self.env.step(action, False)
            if episode.time_steps > max_steps :
                done = True
                reward = MAX_STEPS_REWARD
            
            G, node_inputs, leaf_nodes = state  
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])

            state = next_state
            command[0] = min(command[0]-reward, MAX_REWARD) # desired reward
            command[1] = max(command[1]-1, 1)               # desired time frame
            steps += 1
            total_reward += reward
            self.logger.log_step(reward, loss)
        
        self.logger.log_episode()
        self.logger.record(episode=self.current_iteration)
        self.current_iteration += 1
        
        print("Ended episode | Done : ", done, " Steps : ", steps, " Reward :", total_reward, " Loss: ", loss)
        return episode

    def train_batch(self, steps:int, batch_size=10):
        loss_total = 0
        print("Started training")
        for step in range(steps):
            episodes = self.memory.sample(min(self.mem_size, batch_size))
            for episode in episodes:
                graphs, inputs, actions, leaves, command = episode.sample()
                new_predictions = self.behaviour(graphs, inputs, leaves, command)
                loss = F.cross_entropy(new_predictions, actions.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
            print("  Update Complete :", step)
        print("Training batch complete")
        return loss_total

    def run_udrl(self, iterations=100, fileName="firstRun.pt"):
        for i in range(iterations):
            episode = self.run_episode(policy=BEHAVIOUR_ACT)
            self.memory.add_episode(episode)


trainer = Trainer(device='cuda')
trainer.run_udrl()



# chnages

# removed sigmoid from the embedding function