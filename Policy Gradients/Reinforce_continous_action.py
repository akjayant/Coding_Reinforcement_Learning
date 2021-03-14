#------------------THIS IS VANILLA REINFORCE ALGORITHM WITH NO BASELINE-----------------------------------------------

import gym
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from gym import wrappers
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import safety_gym

from torch.distributions import MultivariateNormal


class PolicyNetwork(nn.Module):
    def __init__(self,num_states, num_actions, hidden_size):
        super(PolicyNetwork,self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        #Policy Network
        self.actor = nn.Sequential(
        nn.Linear(num_states,hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size,hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size,2*num_actions)
        )


    def forward(self,state):
        nn_out = self.actor(state).view(-1,4)


        mu_1 = T.tanh(nn_out[0][0]).view(-1,1)
        var_1 = F.softplus(nn_out[0][1]).view(-1,1)


        mu_2 = T.tanh(nn_out[0][2]).view(-1,1)
        var_2 = F.softplus(nn_out[0][3]).view(-1,1)
        #print(mu_1,mu_2)
        #print(var_1,var_2)
        mu = T.cat([mu_1,mu_2]).to(device)
        var = T.cat([var_1,var_2]).to(device)

        mu = mu.view(-1,1,2)
        var = var.view(-1,1,2)
        #var = T.ones_like(var)
        #var = var*0.1
        dist = MultivariateNormal(mu,T.diag_embed(var))

        return dist




def update_gradients(gamma,ep_rewards,ep_logits,ep_entropies):
    mc_return = []
    p_loss = []
    loss=0
    G = 0
    for r in reversed(range(len(ep_rewards))):
        G = ep_rewards[r] + gamma*G
        mc_return.insert(0,G)
    mc_return = torch.tensor(mc_return)
    advantage_returns = (mc_return - mc_return.mean())/mc_return.std()
    #print((mc_return))
    for lp, re in zip(ep_logits, advantage_returns):
        p_loss.append( - lp * re)

    optim_policy.zero_grad()
    #trying entropy regularization
    #print(ep_entropies)
    loss = torch.stack(p_loss).sum() #+ 0.0001*ep_entropies
    loss.backward()
    #plot_grad_flow_v2(p_net.named_parameters())
    optim_policy.step()



def train(env):
    gamma = 0.99
    max_episodes = 1400
    max_steps = 1000
    running_reward = 0
    running_cost = 0
    plot_rewards = []
    plot_costs = []
    for ep in range(max_episodes):
        ep_rewards =[]
        ep_logits = []
        ep_entropies = []
        ep_costs = []
        current_reward = 0
        current_cost = 0
        state = env.reset()

        for step in range(max_steps):
            #print(type(state))
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            dist_obj = p_net.forward(state)
            #print(mu_out,var_out)
            #Sample next action according normal distribution we trying to fit

            sampled_action_tensor = dist_obj.sample()
            sampled_action = np.clip(sampled_action_tensor.cpu().detach().numpy(),-1,1)
            #print(sampled_action,"/n")
            next_state,reward,done,info= env.step(sampled_action)
            log_prob = dist_obj.log_prob(sampled_action_tensor)
            #print(log_prob)
            entropy = dist_obj.entropy()
            ep_rewards.append(reward)
            ep_logits.append(log_prob)
            ep_entropies.append(entropy)
            current_reward += reward
            current_cost += info['cost']
            state = next_state
            running_reward = 0.05 * current_reward + (1 - 0.05) * running_reward
            running_cost = 0.05 * current_cost + (1 - 0.05) * running_cost

            if done:
                plot_rewards.append(current_reward)
                plot_costs.append(current_cost)
                break
        if ep%10==0:
            print(ep)
            print("Current reward = ",current_reward,"Running reward = ",running_reward,"Current cost = ",current_cost,"Running cost = ",running_cost)
        #Update the parameters
        ep_entropies = torch.cat(ep_entropies)
        update_gradients(gamma,ep_rewards,ep_logits,ep_entropies.sum())
        writer.add_scalar("Reward ",current_reward)
        writer.add_scalar("Cost ",current_cost)
        #if running_reward >env.spec.reward_threshold:
        #    print("Solved in ",ep)
    return plot_rewards, dist_obj, plot_costs



    #device = set_device()
env = gym.make('Safexp-PointGoal1-v0')
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print(device)
p_net = PolicyNetwork(60,2,256)
p_net.to(device)
optim_policy= optim.Adam(p_net.parameters(), lr=3e-4)
writer = SummaryWriter()
p_net.train()


plot_rewards, dist_obj, plot_costs = train(env)
writer.flush()
env.close()

cost_limit = np.array([25 for i in range(1400)])


fig, axs = plt.subplots(2)
# axs[0].plot(x, running_avg_return)
# axs[1].plot(x, running_avg_cost)
# axs[1].plot(x,limit_cost)
# axs[0].set_yticks(np.arange(0,26,5))
# axs[1].set_yticks(np.arange(0,161,20))
# plt.savefig(figure_file)

#axs[0].plot(np.arange(0,1400),plot_rewards)
#axs[1].plot(np.arange(0,1400),plot_costs)
axs[0].plot(np.arange(0,1400),pd.Series(plot_rewards).rolling(100).mean())
axs[1].plot(np.arange(0,1400),pd.Series(plot_costs).rolling(100).mean())
axs[1].plot(np.arange(0,1400),cost_limit)
axs[0].set_yticks(np.arange(0,26,5))
axs[1].set_yticks(np.arange(0,161,20))
plt.savefig('vpg_pointgoal1.png')
