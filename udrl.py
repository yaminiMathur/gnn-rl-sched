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

cuda = args.cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        print("Buffer initialized... ")
        
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

        

class MetricLogger():
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
        self.ep_avg_qs_plot = save_dir + "/q_plot_"+aggregator+"_"+version+".png"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 6)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 6)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 6)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 6)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode :  {episode} \n"
            f"Step :  {step} \n"
            f"Epsilon :  {epsilon} \n"
            f"Mean Reward :  {mean_ep_reward} \n"
            f"Mean Length :  {mean_ep_length} \n"
            f"Mean Loss :  {mean_ep_loss} \n"
            f"Mean Q Value :  {mean_ep_q} \n"
            f"Time Delta :  {time_since_last_record} "
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.xlabel('episodes')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()




# He has a class for behaviour -> behaviour is generally defined by a neural network.
# He has multiple steps to calculate state and command embeddings
# Which he passes through an output neural network
# In our case, we have an Net defining just the neural network, This will have the forward part 
# Here define what you want to do for state and command like what he has done
# In out case agent will have the action, greedy action, load, save, and udrl
# Please figure out what each line does --- Comment on every line if needed, google package name and see what it is used for
# if you don't understand
# Correlate what he has done with what You should. If you feel that his method is easier --> go with it ( and follow only one convention)