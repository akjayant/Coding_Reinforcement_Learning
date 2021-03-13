import torch as T
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple
import os
class Critic(nn.Module):
    def __init__(self,beta,input_dims,fc1_dims,fc2_dims, n_actions, name, chckpt_dir='models/'):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chckpt_dir,name+'ddpg_critic')

        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        T.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #when using batch_norms we have to use train() and eval()
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        T.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims,1)
        T.nn.init.uniform_(self.q.weight.data,-f3,f3)
        T.nn.init.uniform_(self.q.bias.data,-f3,f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = torch.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = torch.relu(self.action_value(action))
        state_action_value = T.relu(T.add(state_value,action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value
    def save_checkpoint(self):
        print("checkpointing....")
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Actor(nn.Module):
    def __init__(self,alpha,input_dims, fc1_dims,fc2_dims, n_actions, name, chckpt_dir='models/'):

        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chckpt_dir,name+'ddpg_actor')
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        T.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        T.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims,self.n_actions)
        T.nn.init.uniform_(self.fc2.weight.data,-f3,f3)
        T.nn.init.uniform_(self.fc2.bias.data,-f3,f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self,state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = T.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = T.relu(x)
        x = T.tanh(self.mu(x))

        return x
    def save_checkpoint(self):
        print("checkpointing....")
        T.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
