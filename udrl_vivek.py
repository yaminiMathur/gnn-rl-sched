### Import Libraries
from typing import List, Tuple
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from environment_wrapper import *
from dgl.nn.pytorch import SAGEConv
import numpy as np
import datetime
from matplotlib import pyplot as plt
import pickle

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

    # Compute the action from a state and command
    # Note: this is only for computing a single action and not for training
    def action(self, state, command):
        G, node_inputs, leaf_nodes = state
        predictions = self.forward(state, command)
        return leaf_nodes[predictions.argmax().item()]

    # Overloaded forward function
    def forward(self, state, command):
        G, node_inputs, leaf_nodes = state
        gnn_embeddings = self.gcn(G.to(self.device), node_inputs.to(self.device))
        encoded_command = self.command_fn(torch.FloatTensor(command).to(self.device))
        product = torch.mul(gnn_embeddings[leaf_nodes], encoded_command)
        return self.output_nn(product)

    # Get the random action for state
    def random_action(self, env:GraphWrapper, assist_probability=50):
        G, node_inputs, leaf_nodes = env.observe()
        action = leaf_nodes[(random.randint(0, len(leaf_nodes)-1))]
        if random.randint(0, 100-1) < assist_probability :
            action = env.auto_step()
        return action

    # save the model parameters
    def save(self, filename):
        torch.save(self.state_dict(), "./models/behaviour_"+filename)
    
    # load model parameters
    def load(self, filename):
        self.state_dict(torch.load("./models/behaviour_"+filename))

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

    def sample(self):
        total_len = len(self.list)
        start = random.randint(0, total_len-2); end = random.randint(start+1, total_len-1)
        horizon = end-start
        G, node_inputs, leaf_nodes, action, reward = self.list[start]
        tot_rewards = sum([observation[4] for observation in self.list[start:end+1]])
        actions = torch.FloatTensor([int(action == leaf) for leaf in leaf_nodes]).reshape(1, len(leaf_nodes))
        return G, node_inputs, actions, leaf_nodes, [tot_rewards, horizon]

    def __lt__(self, other):
        return self.total_reward > other.total_reward

    def __len__(self):
        return len(self.list)

### Replay Buffer for episodes
class Memory:

    def __init__(self, load_file=None, load_version=0):
        
        self.buffer = []
        if load_file:
            print("Loading warmup data file..")
            self.load(load_file, load_version)
        
    def add_episode(self, episode:Episode):
        self.buffer.append(episode)
        self.sort()
    
    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)
        return np.random.choice(self.buffer, 
            size=min(batch_size, len(self.buffer)-1), replace=False)

    def get_possible_command(self, count=10):
        rewards = []
        steps = []
        for episode in self.buffer[-(min(count, len(self.buffer))):]:
            rewards.append(episode.total_reward)
            steps.append(episode.time_steps)
        
        possible_horizon = np.mean(steps)
        rewards_std = min(np.std(rewards), 1)
        rewards_mean = np.mean(rewards)
        possible_reward = np.random.uniform(rewards_mean, rewards_mean+rewards_std)

        return [possible_reward, possible_horizon]
         
    def sort(self):
        self.buffer.sort()
    
    def load(self, data_file, load_version):
        print("loading training data")
        with open("./warmup/"+data_file+"_"+str(load_version)+".dat", "rb") as f:
            self.buffer = pickle.load(f)
        
    def save(self, version):
        print("Saving Training Data")
        with open("./warmup/warmup_"+str(version)+".dat", "wb") as f:
            pickle.dump(self.buffer, f)

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------------------------------------------

class MetricLogger:

    def __init__(self, save_dir="./results", mode="train", version="0", aggregator="mean"):
        save_dir = save_dir+"/"+mode
        self.save_log = save_dir + "/episodes_"+aggregator+"_"+version+".log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot_"+aggregator+"_"+version+".png"
        self.ep_lengths_plot = save_dir + "/length_plot_"+aggregator+"_"+version+".png"
        self.ep_avg_losses_plot = save_dir + "/loss_plot_"+aggregator+"_"+version+".png"
        self.episode = 1

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
        self.curr_ep_loss += loss

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.episode += 1
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0

    def record(self):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-10:]), 6)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-10:]), 6)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-10:]), 6)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode :  {self.episode} \n"
            f"Mean Reward :  {mean_ep_reward} \n"
            f"Mean Length :  {mean_ep_length} \n"
            f"Mean Loss :  {mean_ep_loss} \n"
            f"Time Delta :  {time_since_last_record} "
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{self.episode:8d}{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.3f}{time_since_last_record:15.3f}"
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

    def __init__(self, device, optimizer=Adam, lr=0.00001, aggregator="pool", 
                 version=0, assist=100, nn_file=None, data_file=None, 
                 episode_stop=10000, data_version=0, warmup_ep=10) -> None:
        
        # initialize the objects required
        self.memory = Memory(data_file, data_version)
        self.behaviour = Behaviour(device=device, aggregator=aggregator)
        self.logger = MetricLogger(aggregator=aggregator, version=str(version))
        self.env = GraphWrapper()
        self.loss_fn = nn.CrossEntropyLoss()
        
        # if file is given as arg then load trained weights
        if nn_file:
            self.behaviour.load(nn_file)
            
        self.optimizer = optimizer(self.behaviour.parameters(), lr=lr)
        self.device = device
        self.aggregator = aggregator
        self.version = version
        self.episode_stop = episode_stop
        self.assist_probability = assist
        
        # load generated training data if file is already present
        if not data_file:
            self.generate_warmup_data(warmup_ep, data_version)
            
    def generate_warmup_data(self, warmup_ep, data_version) :
        print("Generating warmup data")
        for i in range(warmup_ep):
            
            # init episode details
            episode = Episode(init_command=[1, 1])
            self.env.reset(random.randint(1, 100000))
            done = False
            
            # run the episode with random actions
            while not done and episode.time_steps < self.episode_stop:
                state = self.env.observe() 
                index = self.behaviour.random_action(self.env, self.assist_probability)
                next_state, reward, done = self.env.step((index, 1), False)
                if done:
                    reward = 50
                G, node_inputs, leaf_nodes = state
                episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])

            # add episode to memory
            print("Ended episode | Done : ", done, " | Steps : ", episode.time_steps, " | Reward :", episode.total_reward)
            self.memory.add_episode(episode)
        
        self.memory.save(str(data_version))

    def run_episode(self, iteration, command=[1, 1], seed=None, train_params={"steps":5, "batch": 5}):
        
        # init seed if not initialized already
        if not seed :
            seed = random.randint(1, 100000)

        # init the episode params
        episode = Episode(init_command=command)
        self.env.reset(seed)
        state = self.env.observe()
        self.logger.init_episode()
        done = False; loss = 0
        
        while not done and  episode.time_steps < self.episode_stop:
            
            loss = self.train_batch(steps=train_params["steps"], batch_size=train_params["batch"])
            index = self.behaviour.action(state, command)
            next_state, reward, done = self.env.step((index, 1), False)
            
            if done:
                reward = 50
            elif episode.time_steps > self.episode_stop :
                done = True
                reward = MAX_STEPS_REWARD
            
            G, node_inputs, leaf_nodes = state  
            episode.add_iteration([G, node_inputs, leaf_nodes, index, reward])
            
            state = next_state                              # next state
            command[0] = min(command[0]-reward, MAX_REWARD) # desired reward
            command[1] = max(command[1]-1, 1)               # desired time frame

            self.logger.log_step(reward, loss)
            # if episode.time_steps % 293 == 0:
            #     self.logger.log_episode()
            #     self.logger.record()
        
        print("Ended episode | Done : ", done, " | Steps : ", episode.time_steps, " | Reward :", episode.total_reward, " | Loss: ", self.logger.curr_ep_loss)
        self.logger.log_episode()
        self.logger.record()
        self.behaviour.save("scheduling_"+str(self.version)+"_"+str(iteration)+"_"+self.aggregator+".pt")
        
        return episode

    def train_batch(self, steps=1, batch_size=5):
        loss_total = 0
        for step in range(steps):
            episodes = self.memory.sample(batch_size)
            for episode in episodes:
                graphs, inputs, actions, leaves, command = episode.sample()
                self.optimizer.zero_grad()
                predictions = self.behaviour((graphs, inputs, leaves), command)
                loss = self.loss_fn(predictions.reshape(1, len(leaves)), actions.to(self.device))
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
        return loss_total

    def run_udrl(self, iterations=100, command_batch=5):
        for i in range(iterations):
            command = self.memory.get_possible_command(count=command_batch)
            episode = self.run_episode(i, command=command.copy())
            self.memory.add_episode(episode)

trainer = Trainer(device='cuda', lr=0.00001, aggregator="mean", 
                  version=6, assist=100, nn_file=None,
                 data_file="warmup", episode_stop=10000, data_version=0, warmup_ep=10)

trainer.run_udrl(iterations=20, command_batch=5)