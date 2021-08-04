import torch
import torch.nn as nn
import torch.nn.functional as F 

class Actor(nn.Module):

    def __init__(self, action_space=100, num_inputs=500):
        super(Actor, self).__init__()

        # mu.weight.data.mul_(0.1)
        # mu.bias.data.mul_(0.1)

        self.l1    = nn.Linear(num_inputs, 64)
        self.ln1   = nn.LayerNorm(64)

        self.l2    = nn.Linear(64, 64)

        self.l3    = nn.Linear(64, 64)            
        self.ln3   = nn.LayerNorm(64)

        self.l4    = nn.Linear(64, action_space)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.ln1(x)
        x = torch.sigmoid(x)

        x = self.l2(x)
        x = torch.sigmoid(x)

        x = self.l3(x)
        x = self.ln3(x)
        x = torch.sigmoid(x)

        x = self.l4(x)
        
        return x

class Critic(nn.Module):

    def __init__(self,  action_space=200, num_inputs=500):
        super(Critic, self).__init__()

        self.l1    = nn.Linear(num_inputs, 64)
        self.ln1   = nn.LayerNorm(64)

        self.l2    = nn.Linear(64+action_space, 64)
        self.ln2   = nn.LayerNorm(64)

        self.l3    = nn.Linear(64, 32)
        self.l4    = nn.Linear(32, 16)
        self.ln4   = nn.LayerNorm(16)

        self.V     = nn.Linear(16, 1)

    def forward(self, inputs, actions):
        
        x = self.l1(inputs)
        x = self.ln1(x)
        x = torch.sigmoid(x)

        x = torch.cat((x, actions))
        x = self.l2(x)
        x = self.ln2(x)
        x = F.sigmoid(x)

        x = self.l3(x)
        x = torch.sigmoid(x)

        x = self.l4(x)
        x = self.ln4(x)
        x = torch.sigmoid(x)

        V = self.V(x)
        
        return V

class ActorCritic(nn.Module):
    def __init__(self, action_node=50, action_exec=50, num_inputs=100):
        super().__init__()
        
        self.actor_node          = Actor(action_space=action_node, num_inputs=num_inputs)
        self.actor_parallelism   = Actor(action_space=action_exec, num_inputs=num_inputs)
        self.critic_node         = Critic(action_space=action_node+action_exec, num_inputs=num_inputs)
        self.critic_parallelism  = Critic(action_space=action_node+action_exec, num_inputs=num_inputs)
        
    def forward(self, state):
        
        action_node        = self.actor_node(state)
        action_parallelism = self.actor_parallelism(state)
        value_node_pred    = self.critic_node(state, torch.cat([action_node, action_parallelism.detach()]))
        value_para_pred    = self.critic_parallelism(state, torch.cat([action_node.detach(), action_parallelism]))
        
        return action_node, action_parallelism, value_node_pred, value_para_pred

