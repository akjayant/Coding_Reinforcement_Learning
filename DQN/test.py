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

def torchify_state_dim(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state).float()
    return state.unsqueeze(0)

def test(env, n_episodes, policy,device, render=False):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = torchify_state_dim(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = torchify_state_dim(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    agent = torch.load("saved_model")
    test(env, 1, agent,device)
