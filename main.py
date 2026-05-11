from itertools import accumulate

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box


def mlp(obs_dim, n_actions):
    return nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), 
                         nn.Linear(32, n_actions), nn.Identity())

def train(lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make("CartPole-v1")
    demo_env = gym.make("CartPole-v1", render_mode="human") if render else None

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = mlp(obs_dim, n_actions)
    
    def get_policy(obs):
        return Categorical(logits=model(obs))
    
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def compute_loss(obs, act, weights):
        return -(get_policy(obs).log_prob(act)*weights).mean()

    optimizer = Adam(model.parameters(), lr=lr)

    def run_demo_episode():
        obs, _ = demo_env.reset()
        done = False
        while not done:
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, _ = demo_env.step(act)
            done = terminated or truncated
    
    def reward_to_go(rews):
        return list(accumulate(reversed(rews)))[::-1]

    def epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs, _ = env.reset()
        done = False
        ep_rews = []

        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, _ = env.reset()
                done = False
                ep_rews = []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    
    for i in range(epochs):
        if render:
            run_demo_episode()
        batch_loss, batch_rets, batch_lens = epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    train(render=True)
