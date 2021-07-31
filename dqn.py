import copy
from collections import deque
import random
from networkx.readwrite import leda
import numpy as np
import torch
import torch.nn as nn
import time, datetime
from matplotlib import pyplot as plt
from dgl.nn.pytorch import GraphConv
from numpy.random import randint
import dgl

device = "cuda"

class GCN(nn.Module):

    def __init__(self, features=5, hidden_layer_size=5, embedding_size=1):
        super(GCN, self).__init__()
        
        self.conv1 = GraphConv(in_feats=features, out_feats=hidden_layer_size)
        self.conv2 = GraphConv(in_feats=hidden_layer_size, out_feats=embedding_size)
        # self.conv3 = GraphConv(hidden_layer_size, embedding_size)
        self.softmax = nn.Softmax()

    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h)
        h = torch.sigmoid(h)
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        # h = self.softmax(h)

        return h

class Net(nn.Module):

    def __init__(self, features=5, hidden_layer_size=5, embedding_size=1):
        super().__init__()
        
        # The GNN for online training and the target which gets updated 
        # in a timely manner
        self.online = GCN(features, hidden_layer_size, embedding_size)
        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, graph, node_input, model="online"):
        if model == "online":
            logits = self.online(graph, node_input)
        else :
            logits = self.target(graph, node_input)

        return logits  

class Agent():

    def __init__(self, save_dir="./models", assist=True, assist_p=(1, 7)):
        self.save_dir = save_dir

        # DNN to predict the most optimal action
        self.net = Net().float()
        self.net = self.net.to(device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.save_every = 5e3  # no. of experiences

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4      # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.assist = assist
        self.assist_range = assist_p
        self.softmax = nn.Softmax()

    def act(self, state, parallelism=1):
        
        G, node_inputs, leaf_nodes = state
        
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            index = randint(len(leaf_nodes))
            if self.assist:
                action_idx = leaf_nodes[index]
            else :
                action_idx = leaf_nodes[index]

        # EXPLOIT
        else:
            logits = self.net(G, node_inputs, model="target")
            req    = torch.argmax(logits[leaf_nodes]).item()
            action_idx = leaf_nodes[req]


        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        reward = torch.tensor([reward]).to(device)
        done = torch.tensor([done]).to(device)
        action = torch.tensor(action).to(device)
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        return random.sample(self.memory, self.batch_size)
        # state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        # return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        state, node_inputs, leaf_nodes = state
        current_Q = self.net(state, node_inputs, model="online")[action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):

        next_state, node_inputs, leaf_nodes = next_state

        if len(leaf_nodes) == 0:
            return reward

        logits = self.net(next_state, node_inputs, model="online")
        req    = logits[leaf_nodes]
        index  = torch.argmax(req).item()
        best_action = leaf_nodes[index]
        
        next_Q = self.net(next_state, node_inputs, model="target")[best_action]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def td_estimate_batch(self, state, action):
        # actions need to be re indexed based on batching
        state, node_inputs = state
        current_Q = self.net(state, node_inputs, model="online")[action]
        return current_Q

    @torch.no_grad()
    def td_target_batch(self, reward, next_state, done):

        # leaf_nodes need to be re constructed as (ending_index, leaf_nodes)
        next_state, node_inputs, leaf_nodes = next_state
        indices = []; action_values = []
        prev = 0; reward_indices = 0

        # calculate the logits for online model
        logits = self.net(next_state, node_inputs, model="online")
        logits = logits.detach()

        # seperate the actions per graph
        for i, leaves in leaf_nodes:
            # if len(leafs) == 0:
            #     reward_indices += 1
            #     prev = i
            #     continue
            req = logits[prev:i, :][leaves]
            index = torch.argmax(req).item()
            indices.append(leaves[index]+prev)
            prev = i
            action_values.append(reward_indices)
            reward_indices += 1

        # calculate the next Q values
        next_Q = self.net(next_state, node_inputs, model="target")[indices]
        # values = torch.zeros(len(reward)).to(device)
        # values[action_values] = next_Q.flatten()
        # values = values.reshape(32, 1)

        return (reward + (1 - done.float()) * self.gamma * next_Q).float() # values).float()

    def update_Q_batch(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.sum()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.target.state_dict())

    def save(self):
        save_path = (
            self.save_dir + f"/sched_net_{int(self.curr_step // self.save_every)}.pt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Sched_net saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        memory = self.recall()
        estimate = []
        loss_list = []
        
        for sample in memory:
            state, next_state, action, reward, done = sample

            if len(state[2]) == 0:
                continue

            # Get TD Estimate
            td_est = self.td_estimate(state, action)

            # Get TD Target
            td_tgt = self.td_target(reward, next_state, done)

            # Backpropagate loss through Q_online
            loss_list.append(self.update_Q_online(td_est, td_tgt))
            estimate.append(td_est)

        return (torch.tensor(estimate).flatten().mean().item(), torch.tensor(loss_list).flatten().sum())

    def get_batch(self):
        # Sample from memory
        memory = self.recall()

        length_current = 0; length_next = 0 
        graphs = []; node_input_list = []; actions = []; rewards = []; done_list = []
        next_graphs = []; next_leaf_nodes = []; next_node_inputs = []

        for sample in memory:
            state, next_state, action, reward, done = sample

            # unpack the state
            graph, node_inputs, leaf_nodes = state
            next_graph, next_node_input, next_leaves = next_state

            # actions have to be appended before length is increased
            actions.append(action + length_current)
            length_current += graph.number_of_nodes()

            # append values to respective lists to create batches
            graphs.append(graph)
            node_input_list.append(node_inputs)
            rewards.append(reward)
            done_list.append(done)

            # set the length of the nodes to be used for next_state
            length_next += next_graph.number_of_nodes()

            # append values to lists to create next_state
            next_graphs.append(next_graph)
            next_node_inputs.append(next_node_input)
            next_leaf_nodes.append((length_next, next_leaves))

        # create tensors to train as batches
        batch_state = (dgl.batch(graphs).to(device), torch.cat(node_input_list).to(device))
        batch_actions = torch.stack(actions).to(device)
        batch_rewards = torch.stack(rewards).to(device)
        batch_next_state = (dgl.batch(next_graphs).to(device), torch.cat(next_node_inputs).to(device), next_leaf_nodes)
        batch_done = torch.stack(done_list)

        return batch_state, batch_next_state, batch_actions, batch_rewards, batch_done

    def learn_batch(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, actions, rewards, done = self.get_batch()

        # Get TD Estimate       
        td_est = self.td_estimate_batch(state, actions)

        # Get TD Target
        td_tgt = self.td_target_batch(rewards, next_state, done)
        
        # get the loss function
        loss = self.update_Q_batch(td_est, td_tgt)

        return td_est.flatten().mean().item(), loss

class MetricLogger:
    def __init__(self, save_dir="./results"):
        self.save_log = save_dir + "/episodes.log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot.png"
        self.ep_lengths_plot = save_dir + "/length_plot.png"
        self.ep_avg_losses_plot = save_dir + "/loss_plot.png"
        self.ep_avg_qs_plot = save_dir + "/q_plot.png"

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
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
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
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()