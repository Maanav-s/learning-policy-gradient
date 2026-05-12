import argparse
import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Discrete, Box


CHECKPOINT_DIR = "checkpoints"
FIGURES_DIR = "figures"
ENV_ID = "LunarLander-v3"
ENV_KWARGS = dict(
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

class Policy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dims, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, act_dims)
        self.log_stds = nn.Parameter(torch.zeros(act_dims))
    
    def forward(self, obs):
        base = self.base(obs)
        m = self.mean_head(base)
        std = self.log_stds.exp()
        return torch.distributions.Normal(m, std)

    def act(self, obs):
        dist = self.forward(obs)
        u = dist.rsample()
        log_prob = dist.log_prob(u).sum(-1)

        action = torch.tanh(u)
        log_prob -= (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1) # Apparently this adds numerical stability

        return action, log_prob


def train(lr=1e-3, epochs=25, batch_size=90000, gamma=0.985, render=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = gym.make(ENV_ID, **ENV_KWARGS)
    demo_env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS) if render else None

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    
    def compute_loss(log_prob, weights):
        return -(log_prob*weights).mean()

    optimizer = Adam(model.parameters(), lr=lr)

    def run_demo_episode():
        obs, _ = demo_env.reset()
        done = False
        while not done:
            act, _ = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, _ = demo_env.step(act.detach().numpy())
            done = terminated or truncated
    
    def reward_to_go(rews):
        out, running = [], 0.0
        for r in reversed(rews):
            running = r + gamma * running
            out.append(running)
        return out[::-1]

    def epoch():
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
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
            act, log_probs = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act.detach().numpy())
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            batch_log_probs.append(log_probs)
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
        weights = torch.as_tensor(batch_weights, dtype=torch.float32)
        weights = (weights - weights.mean()) / (weights.std())
        batch_loss = compute_loss(torch.stack(batch_log_probs), weights)
        batch_loss.backward()
        optimizer.step()
        return batch_loss.item(), batch_rets, batch_lens
    
    losses, mean_returns, mean_ep_lens = [], [], []
    for i in range(epochs):
        if render:
            run_demo_episode()
        batch_loss, batch_rets, batch_lens = epoch()
        losses.append(batch_loss)
        mean_returns.append(float(np.mean(batch_rets)))
        mean_ep_lens.append(float(np.mean(batch_lens)))
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, mean_returns[-1], mean_ep_lens[-1]))

    ckpt_path = os.path.join(CHECKPOINT_DIR, "policy.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    save_training_figures(losses, mean_returns, mean_ep_lens)


def save_training_figures(losses, mean_returns, mean_ep_lens):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    epochs_axis = range(1, len(losses) + 1)

    for name, values, ylabel in [
        ("loss", losses, "policy loss"),
        ("return", mean_returns, "mean episode return"),
        ("episode_length", mean_ep_lens, "mean episode length"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(epochs_axis, values)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per epoch")
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure to {out_path}")


def demo(checkpoint_path, n_episodes=5):
    env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = Policy(obs_dim, act_dim)
    model.load_state_dict(torch.load(checkpoint_path))

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                act, _ = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act.numpy())
            total += rew
            done = terminated or truncated
        print(f"episode {ep+1}: return = {total:.2f}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "demo"])
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint (required for demo)")
    parser.add_argument("--episodes", type=int, default=5, help="number of demo episodes")
    parser.add_argument("--render", action="store_true", help="render an episode each epoch during training")
    args = parser.parse_args()

    if args.mode == "train":
        train(render=args.render)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for demo mode")
        demo(args.checkpoint, n_episodes=args.episodes)
