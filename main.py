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
    gravity=-9.8,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
)

class Policy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dims, 64), nn.LayerNorm(64), nn.Tanh(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.Tanh(),
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

        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

class Critic(nn.Module):
    def __init__(self, obs_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dims, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def train(lr=3e-3, batches=100000, batch_step=25, gamma=0.985, entropy_coef=0.001, render=False, lam=0.95):
    envs = gym.make_vec(ENV_ID,
                           num_envs = 25,
                           vectorization_mode="sync",
                           **ENV_KWARGS)
    demo_env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS) if render else None

    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    critic = Critic(obs_dim)
    actor_optimizer = Adam(model.parameters(), lr=lr)
    critic_optimizer = Adam(critic.parameters(), lr=lr)

    def gae(rews, dones, values, gamma, l): #l is lambda
        rews = np.asarray(rews, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        advantages = np.empty_like(rews)
        running = np.zeros(rews.shape[1:], dtype=np.float32)
        for t in range(rews.shape[0] - 1, -1, -1):
            nonterminal = (1.0 - dones[t])
            deltas = rews[t] - values[t] + gamma * values[t+1] * nonterminal
            running = deltas + gamma * l * running * nonterminal
            advantages[t] = running
        return advantages

    def run_demo_episode():
        obs, _ = demo_env.reset()
        done = False
        while not done:
            act, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, _ = demo_env.step(act.detach().numpy())
            done = terminated or truncated

    T, N = batch_step, envs.num_envs
    obs_buf  = np.zeros((T, N, obs_dim), dtype=np.float32)
    rew_buf  = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    logp_buf = [None] * T
    ent_buf  = [None] * T

    ep_returns_log = []
    ep_lengths_log = []
    ep_ret_running = np.zeros(N, dtype=np.float32)
    ep_len_running = np.zeros(N, dtype=np.int64)

    obs, _ = envs.reset()

    def run_batch(obs):
        nonlocal ep_ret_running, ep_len_running
        for t in range(T):
            obs_buf[t] = obs
            act, log_prob, entropy = model.act(torch.as_tensor(obs, dtype=torch.float32))
            logp_buf[t] = log_prob
            ent_buf[t]  = entropy
            obs, rew, term, trunc, _ = envs.step(act.detach().numpy())
            rew_buf[t]  = rew
            done = term | trunc
            done_buf[t] = done

            ep_ret_running += rew
            ep_len_running += 1
            for i in np.where(done)[0]:
                ep_returns_log.append(float(ep_ret_running[i]))
                ep_lengths_log.append(int(ep_len_running[i]))
            ep_ret_running[done] = 0.0
            ep_len_running[done] = 0

        flat_obs = torch.as_tensor(obs_buf.reshape(T * N, obs_dim), dtype=torch.float32)
        values = critic(flat_obs)                                  # (T*N,) with grad

        # bootstrap V(s_T) on the obs returned after the last step, then build (T+1, N) for GAE
        last_values = critic(torch.as_tensor(obs, dtype=torch.float32))       # (N,)
        values_for_gae = np.concatenate([
            values.detach().numpy().reshape(T, N),
            last_values.detach().numpy()[None, :],
        ], axis=0)                                                 # (T+1, N)

        advantage_np = gae(rew_buf, done_buf, values_for_gae, gamma, lam)     # (T, N)
        advantage_raw = torch.as_tensor(advantage_np.reshape(-1), dtype=torch.float32)
        returns = advantage_raw + values.detach()                  # TD(λ) returns as critic target

        advantage = (advantage_raw - advantage_raw.mean()) / (advantage_raw.std() + 1e-8)

        log_probs = torch.stack(logp_buf).reshape(-1)              # (T*N,)
        entropies = torch.stack(ent_buf).reshape(-1)               # (T*N,)

        actor_loss = -(log_probs * advantage).mean() - entropy_coef * entropies.mean()
        critic_loss = ((returns - values) ** 2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        return obs, actor_loss.item(), critic_loss.item(), rew_buf.sum(0).mean()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "policy.pt")

    actor_losses, critic_losses = [], []
    for b in range(batches):
        obs, a_loss, c_loss, mean_rew = run_batch(obs)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)
        if b % 1000 == 0:
            recent_ret = float(np.mean(ep_returns_log[-50:])) if ep_returns_log else float("nan")
            print(f"batch {b:6d} | actor {a_loss:+.3f} | critic {c_loss:.3f} | window rew {mean_rew:+.3f} | recent ep return {recent_ret:+.3f}")
            if b > 0:
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [periodic] saved checkpoint at batch {b} -> {ckpt_path}")
                if render:
                    run_demo_episode()

    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    save_training_figures(actor_losses, critic_losses, ep_returns_log, ep_lengths_log)


def save_training_figures(actor_losses, critic_losses, ep_returns, ep_lengths):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    def smooth(values, window):
        if len(values) < window or window < 2:
            return np.asarray(values, dtype=np.float32)
        cs = np.cumsum(np.insert(np.asarray(values, dtype=np.float32), 0, 0.0))
        return (cs[window:] - cs[:-window]) / window

    panels = [
        ("actor_loss",     actor_losses,  "actor loss",     "batch"),
        ("critic_loss",    critic_losses, "critic loss",    "batch"),
        ("return",         ep_returns,    "episode return", "completed episode"),
        ("episode_length", ep_lengths,    "episode length", "completed episode"),
    ]
    for name, values, ylabel, xlabel in panels:
        if len(values) == 0:
            print(f"skipping {name}: no data")
            continue
        window = max(1, min(200, len(values) // 20))
        smoothed = smooth(values, window)
        fig, ax = plt.subplots()
        ax.plot(values, alpha=0.25, label="raw")
        if len(smoothed) > 0 and window > 1:
            offset = len(values) - len(smoothed)
            ax.plot(range(offset, len(values)), smoothed, label=f"smoothed (w={window})")
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}")
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="show a live demo episode every 1000 batches")
    args = parser.parse_args()
    train(render=args.render)
