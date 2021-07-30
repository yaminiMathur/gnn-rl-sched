from torch.optim import optimizer
from actor_critic import *
import numpy as np
from numpy.random import randint
from environment_wrapper import *

cuda = args.cuda

class Agent_AC(object):
    def __init__(self, gamma=0.95) -> None:
        super().__init__()
        self.model = ActorCritic().to(cuda)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.saved_actions = []
        self.rewards = []

        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        self.file = "actor_critic"

        self.node_entropy = 0
        self.para_entropy = 0
        self.loss_critic_node = 0
        self.loss_critic_para = 0
        self.loss_node = 0
        self.loss_para = 0

    def learn(self):

        step_reward = 0
        policy_node = [] # list to save actor (policy) loss
        policy_parallelism = []
        value_losses_node = [] # list to save critic (value) loss
        value_losses_para = []
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for reward in self.rewards:
            # calculate the discounted value
            step_reward = reward + self.gamma * step_reward
            returns.insert(0, step_reward)

        returns = torch.tensor(returns).to(cuda)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob_1, log_prob_2, value_node, value_para), reward in zip(self.saved_actions, returns):
            advantage_node = reward - value_node.item()
            advantage_para = reward - value_para.item()

            # calculate actor (policy) loss 
            policy_node.append(-1 * log_prob_1 * advantage_node)
            policy_parallelism.append(-1 * log_prob_2 * advantage_para)

            # calculate critic (value) loss using L1 smooth loss
            value_losses_node.append(F.smooth_l1_loss(value_node, torch.tensor([reward]).to(cuda)))
            value_losses_para.append(F.smooth_l1_loss(value_para, torch.tensor([reward]).to(cuda)))


        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss_node   = torch.stack(policy_node).to(cuda).sum()
        loss_parallelism = torch.stack(policy_parallelism).to(cuda).sum()
        loss_critic_node = torch.stack(value_losses_node).to(cuda).sum()
        loss_critic_para = torch.stack(value_losses_para).to(cuda).sum()

        # perform backprop
        loss = loss_node + loss_critic_node + loss_critic_para + loss_parallelism
        self.loss_node = loss_node
        self.loss_para = loss_parallelism
        self.loss_critic_node = loss_critic_node
        self.loss_critic_para = loss_critic_para
        loss.backward()
        
        self.optimizer.step()

        # reset rewards and action buffer
        self.saved_actions.clear()
        self.rewards.clear()
        
    def act(self, state):
        probs_node, probs_parallelism, value_node, value_para = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        acts_node = torch.distributions.Categorical(logits=probs_node)                      
        acts_parallelism = torch.distributions.Categorical(logits=probs_parallelism) 

        # and sample an action using the distribution
        action_node = acts_node.sample()
        action_parallelism = acts_parallelism.sample()

        self.node_entropy += acts_node.entropy()
        self.para_entropy += acts_parallelism.entropy()

        # save to action buffer
        self.saved_actions.append((acts_node.log_prob(action_node), acts_parallelism.log_prob(action_parallelism), value_node, value_para))

        # the action to take (node, dir) and (parallelism)
        return action_node.item(), action_parallelism.item()

    def save(self, version="1"):
        torch.save(self.model.state_dict(), self.file+"_"+version+".pt")

    def load(self, file="./actor_critic", version="1"):
        self.file = file
        self.model.load_state_dict(torch.load(file+"_"+version+".pt"))
        self.model.eval()

    def direct_train(self, env:EnvironmentWrapper, episodes=70):

        running_reward = 0
        
        for i in range(episodes):
            
            env.reset()
            state = env.observe()
            done = False
            episode_reward = 0

            node_pred = []
            parallelism_pred = []

            node_act = []
            parallelism_act = []
            value_act = []

            while not done:
                
                # get prediction
                node, parallelism, value = self.model(state)

                # get actions from logits for nodes and paralellism limit
                node_pred.append(node); parallelism_pred.append(parallelism)

                # use random actions within bounds to train for action
                n_a = randint(0, 100); p_a = 1
                n = np.zeros(100); p = np.zeros(100); n[n_a] = 1; p[p_a-1] = 1
                
                n_t = torch.from_numpy(n).type(torch.FloatTensor)
                p_t = torch.from_numpy(p).type(torch.FloatTensor)
                node_act.append(n_t)
                parallelism_act.append(p_t)

                # predic value from critic for random actions
                value = self.model.critic( state, torch.cat([n_t.to(cuda), p_t.to(cuda)]) )
                
                # get step and compute actual reward value for random actions
                state, reward, done = env.step(torch.tensor([int(n_a/50), n_a%50, p_a]))
                value_act.append(F.smooth_l1_loss(value, torch.tensor([reward]).type(torch.FloatTensor).to(cuda)))

                episode_reward += reward
            
            n_l = nn.MSELoss()
            p_l = nn.MSELoss()
            loss_node        = n_l(torch.stack(node_pred), torch.stack(node_act).to(cuda))
            loss_parallelism = p_l(torch.stack(parallelism_pred), torch.stack(parallelism_act).to(cuda))
            loss_critic      = torch.stack(value_act).to(cuda).sum()

            total_loss = loss_node + loss_parallelism + loss_critic
            total_loss.backward()

            # loss_node.backward()
            # loss_parallelism.backward()
            # loss_critic.backward()
            self.save()

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # log results
            if i % 5 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i, episode_reward, running_reward))

    def print_metrics(self):
        print("loss_node : {}, loss_para : {}, loss_critic_node : {}, loss_critic_para : {} \n ".format(
            self.loss_node, 
            self.loss_para,
            self.loss_critic_node, 
            self.loss_critic_para)
        )
        self.loss_node = 0
        self.loss_para = 0
        self.loss_critic_node = 0
        self.loss_critic_para = 0