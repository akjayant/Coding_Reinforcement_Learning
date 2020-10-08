import gym
import model
import exp_replay
import random
from wrappers import *
import torch
import torch.nn.functional as F
from collections import namedtuple
from tqdm import tqdm
from itertools import count

class Agent():
    def __init__(self,action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size):
        self.action_space = action_space
        self.frame_history_len = frame_history_len
        self.env = env
        self.device = device
        self.policy_qnet = model.DQN(self.frame_history_len,84,84,6,1e-4).to(self.device)
        self.target_qnet = model.DQN(self.frame_history_len,84,84,6,1e-4).to(self.device)
        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
        self.optimizer = self.policy_qnet.optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = exp_replay.ExperienceReplay(buffer_size)
        self.update_every = update_every
    def epsilon_greedy_act(self, state, eps=0.0):
         #-----epsilon greedy-------------------------------------
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_space)
        else:
            #print(state)


            #print(state)
            #---set the network into evaluation mode(
            self.policy_qnet.eval()
            with torch.no_grad():
                action_values = self.policy_qnet(state.to(self.device))
            #----choose best action
            action = action_values.max(1)[1].view(1,1)
            action = np.argmax(action_values.cpu().data.numpy())
            #----We need switch it back to training mode
            self.policy_qnet.train()
            return action

    def torchify_state_dim(self,obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state).float()
        return state.unsqueeze(0)

    def update_gradients(self):
        gamma = 0.99
        if self.buffer.__len__()<self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        #Preparing batch
        experience = namedtuple('experience',
                        ('state', 'action', 'next_state', 'reward','done'))
        batch = experience(*zip(*batch))
        states = list(map(lambda a: torch.as_tensor(a,device='cuda'),batch.state))
        states = torch.cat(batch.state).to(self.device)
        #print(states.size())
        actions = list(map(lambda a: torch.tensor([[a]],device='cuda'),batch.action))
        actions = torch.cat(actions).to(self.device)
        #print(actions.size())
        rewards = list(map(lambda a: torch.tensor([a],device='cuda'),batch.reward))
        rewards = torch.cat(rewards).to(self.device)
        #print(rewards.size())
        next_states = list(map(lambda a: torch.as_tensor(a,device='cuda'),batch.next_state))
        next_states = torch.cat(next_states).to(self.device)
        #print(list(batch.done))
        #atch.done = [1 if i==True else 0 for i in list(batch.done)]
        dones = list(map(lambda a: torch.tensor([a],device='cuda'),batch.done))
        dones = torch.cat(dones).to(self.device)
        #print(dones.size())
        # Target = r + gamma*(max_a Q_target)
        action_values = self.target_qnet(next_states).detach()
        #print(action_values.max(1)[0].detach())
        max_action_values = action_values.max(1)[0].detach()
        target = rewards + gamma*max_action_values*(1-dones)
        current = self.policy_qnet(states).gather(1,actions)
        target = target.reshape(32,1)
        #print(target.size())
        #print(max_action_values.shape)
        #print(current.size())
        loss = F.smooth_l1_loss(target, current)
        #print("Loss = ",loss)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_qnet.parameters():
            param.grad.data.clamp_(-1.2, 1.2)
        self.optimizer.step()


    def train(self,max_epsiodes,max_steps):
        global steps
        eps = self.epsilon_start
        for episode in tqdm(range(max_epsiodes)):
            obs = self.env.reset()
            state = self.torchify_state_dim(obs)
            total_reward = 0
            for t in  count():
                action = self.epsilon_greedy_act(state,eps)
                next_state,reward,done,_ = self.env.step(action)
                if done:
                    next_state = torch.zeros(state.size())
                    done_flag=1
                else:
                    next_state = self.torchify_state_dim(next_state)
                    done_flag=0
                total_reward += reward
                reward = torch.tensor([reward],device = self.device)

                self.buffer.add(state,action,next_state,reward.to('cpu'),done_flag)
                eps = max(eps * self.epsilon_decay, self.epsilon_min)
                steps += 1
                #print(self.buffer.__len__())
                if steps > 10000:
                    self.update_gradients()
                    if steps%self.update_every==0:
                        self.target_qnet.load_state_dict(self.policy_qnet.state_dict())
                state = next_state
                if done:
                    break
            if episode%10 == 0:
                print("Episode no "+str(episode)+" reward = "+str(total_reward))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = 4
    frame_history_len = 4
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    steps = 0
    buffer_size = 100000
    epsilon_start = 1
    epsilon_decay = 0.99
    epsilon_min = 0.01
    update_every = 1000
    batch_size = 32
    myagent = Agent(action_space,frame_history_len,env,device,buffer_size,\
    epsilon_start,epsilon_decay,epsilon_min,update_every,batch_size)
    myagent.train(500,100000)
    torch.save(myagent.policy_qnet, "saved_model")
    env.close()
