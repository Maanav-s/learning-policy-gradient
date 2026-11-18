import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

CHECKPOINT_DIR = "checkpoints"
FIGURES_DIR = "figures"
VIDEO_DIR = "videos"
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
        unsquashed = dist.rsample()
        log_prob = dist.log_prob(unsquashed).sum(-1)
        log_prob -= (2 * (np.log(2) - unsquashed - nn.functional.softplus(-2 * unsquashed))).sum(-1)
        action = torch.tanh(unsquashed)
        return action, unsquashed, log_prob

    def evaluate(self, obs, unsquashed):
        dist = self.forward(obs)
        log_prob = dist.log_prob(unsquashed).sum(-1)
        log_prob -= (2 * (np.log(2) - unsquashed - nn.functional.softplus(-2 * unsquashed))).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

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

def demo(checkpoint, episodes=10):
    env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            with torch.no_grad():
                dist = model(torch.as_tensor(obs, dtype=torch.float32))
                act = torch.tanh(dist.sample((5,))).mean(dim=0)
            obs, rew, terminated, truncated, _ = env.step(act.numpy())
            ep_return += float(rew)
            ep_length += 1
            done = terminated or truncated
        print(f"episode {ep+1:2d} | return {ep_return:+.2f} | length {ep_length}")

    env.close()

def record(checkpoint,
           episodes=20,
           keep=5,
           seed=0,
           deterministic=True,
           fps=50,
           gif=False,
           min_return=None,
           prefix="lunarlander"):
    """Roll out episodes offscreen, then save the best ones as web-ready mp4 (and optional gif)."""
    import shutil
    import imageio.v2 as imageio

    env = gym.make(ENV_ID, render_mode="rgb_array", **ENV_KWARGS)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    staging = os.path.join(VIDEO_DIR, ".staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    def writer(path):
        return imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=8,
            output_params=["-movflags", "+faststart", "-crf", "20"],
        )

    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        path = os.path.join(staging, f"ep{ep:03d}.mp4")
        with writer(path) as w:
            w.append_data(env.render())
            while not done:
                with torch.no_grad():
                    dist = model(torch.as_tensor(obs, dtype=torch.float32))
                    act = torch.tanh(dist.mean) if deterministic else torch.tanh(dist.sample((5,))).mean(dim=0)
                obs, rew, terminated, truncated, _ = env.step(act.numpy())
                w.append_data(env.render())
                ep_return += float(rew)
                ep_length += 1
                done = terminated or truncated
        results.append(dict(ep=ep, seed=seed + ep, ret=ep_return, length=ep_length, path=path))
        print(f"episode {ep+1:3d}/{episodes} | seed {seed+ep:4d} | return {ep_return:+.2f} | length {ep_length}")

    env.close()

    eligible = [r for r in results if min_return is None or r["ret"] >= min_return]
    if not eligible:
        print(f"\nno episode reached min_return {min_return:+.1f}; nothing saved")
        shutil.rmtree(staging)
        return []
    eligible.sort(key=lambda r: -r["ret"])
    chosen = eligible[:keep]

    saved = []
    for rank, r in enumerate(chosen, start=1):
        base = f"{prefix}_{rank:02d}_return{r['ret']:+.0f}_seed{r['seed']}".replace("+", "p").replace("-", "m")
        mp4_out = os.path.join(VIDEO_DIR, base + ".mp4")
        shutil.move(r["path"], mp4_out)
        print(f"saved {mp4_out}  (return {r['ret']:+.1f}, {r['length']} steps)")
        saved.append(mp4_out)
        if gif:
            gif_out = os.path.join(VIDEO_DIR, base + ".gif")
            frames = imageio.mimread(mp4_out, memtest=False)
            step = max(1, round(fps / 25))
            imageio.mimsave(gif_out, frames[::step], fps=fps / step, loop=0)
            print(f"saved {gif_out}")
            saved.append(gif_out)

    shutil.rmtree(staging, ignore_errors=True)
    rets = np.array([r["ret"] for r in results])
    print(f"\n{len(chosen)} of {episodes} episodes kept | all-episode mean return {rets.mean():+.1f}")
    return saved


def evaluate_policy(checkpoint, episodes=100, seed=0, deterministic=True, threshold=150.0):
    env = gym.make(ENV_ID, **ENV_KWARGS)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    returns = np.zeros(episodes, dtype=np.float64)
    lengths = np.zeros(episodes, dtype=np.int64)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            with torch.no_grad():
                dist = model(torch.as_tensor(obs, dtype=torch.float32))
                act = torch.tanh(dist.mean) if deterministic else torch.tanh(dist.sample((5,))).mean(dim=0)
            obs, rew, terminated, truncated, _ = env.step(act.numpy())
            ep_return += float(rew)
            ep_length += 1
            done = terminated or truncated
        returns[ep] = ep_return
        lengths[ep] = ep_length
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1:3d}/{episodes} episodes | running mean {returns[:ep+1].mean():+.1f}")

    env.close()

    def wilson(successes, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = successes / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    n = len(returns)
    print(f"\n=== {n} evaluation episodes ({'deterministic' if deterministic else 'stochastic'} policy, seeds {seed}..{seed+n-1}) ===")
    print(f"mean return    {returns.mean():+.1f} +/- {returns.std(ddof=1)/np.sqrt(n):.1f} (SEM), std {returns.std(ddof=1):.1f}")
    print(f"median return  {np.median(returns):+.1f}   min {returns.min():+.1f}   max {returns.max():+.1f}")
    print(f"percentiles    p5 {np.percentile(returns,5):+.1f} | p25 {np.percentile(returns,25):+.1f} | p75 {np.percentile(returns,75):+.1f} | p95 {np.percentile(returns,95):+.1f}")
    print(f"mean length    {lengths.mean():.0f} steps")
    for thr in sorted({threshold, 200.0, 0.0}):
        hits = int((returns >= thr).sum())
        lo, hi = wilson(hits, n)
        print(f"return >= {thr:6.0f}  {hits:3d}/{n} = {100*hits/n:5.1f}%   (95% CI {100*lo:.1f}-{100*hi:.1f}%)")

    return returns, lengths


def train(epochs=15000,
          minibatch=125,
          K=4,
          N=25,
          T=25,
          gamma=0.985,
          render=False,
          lam=0.95,
          epsilon=0.2):
    
    envs = gym.make_vec(ENV_ID,
                           num_envs = N,
                           vectorization_mode="sync",
                           **ENV_KWARGS)
    demo_env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS) if render else None

    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.shape[0]

    model = Policy(obs_dim, n_actions)
    critic = Critic(obs_dim)

    def lr_schedule(epoch):
        if epoch < 5000:
            return 3e-3
        return 3e-3 * 10 ** (-(epoch - 5000) / 5000)

    def entropy_schedule(epoch):
        if epoch < 5000:
            return 0.005
        return 0.005 * 0.5 ** ((epoch - 5000) / 2000)

    actor_optimizer = Adam(model.parameters(), lr=lr_schedule(0))
    critic_optimizer = Adam(critic.parameters(), lr=lr_schedule(0))

    def gae(rews, dones, values, gamma, l):
        advantages = np.empty_like(rews)
        running = np.zeros(rews.shape[1:], dtype=np.float32)
        for t in range(rews.shape[0] - 1, -1, -1):
            nonterminal = (1.0 - dones[t])
            delta = rews[t] - values[t] + gamma * values[t+1] * nonterminal
            running = delta + gamma * l * running * nonterminal
            advantages[t] = running
        return advantages

    def run_demo_episode():
        obs, _ = demo_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                act, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, _ = demo_env.step(act.numpy())
            done = terminated or truncated

    batch_size = T * N
    obs_buf  = np.zeros((T, N, obs_dim), dtype=np.float32)
    unsquashed_buf = np.zeros((T, N, n_actions), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf  = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    indices  = np.arange(batch_size)

    ep_returns_log = []
    ep_lengths_log = []
    ep_ret_running = np.zeros(N, dtype=np.float32)
    ep_len_running = np.zeros(N, dtype=np.int64)

    obs, _ = envs.reset()

    def run_epoch(obs, entropy_coef):
        with torch.no_grad():
            for t in range(T):
                obs_buf[t] = obs
                act, unsquashed, logprob = model.act(torch.as_tensor(obs, dtype=torch.float32))
                unsquashed_buf[t] = unsquashed.numpy()
                logp_buf[t] = logprob.numpy()
                obs, rew, term, trunc, _ = envs.step(act.numpy())
                rew_buf[t]  = rew
                done = term | trunc
                done_buf[t] = done

                np.add(ep_ret_running, rew, out=ep_ret_running)
                np.add(ep_len_running, 1, out=ep_len_running)
                for i in np.where(done)[0]:
                    ep_returns_log.append(float(ep_ret_running[i]))
                    ep_lengths_log.append(int(ep_len_running[i]))
                ep_ret_running[done] = 0.0
                ep_len_running[done] = 0

            flat_obs    = torch.as_tensor(obs_buf.reshape(batch_size, obs_dim), dtype=torch.float32)
            values_old  = critic(flat_obs).numpy().reshape(T, N)
            last_values = critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            values_for_gae = np.concatenate([values_old, last_values[None, :]], axis=0)
            advantage_np = gae(rew_buf, done_buf, values_for_gae, gamma, lam)
            returns_np   = advantage_np + values_old

        flat_unsquashed = torch.as_tensor(unsquashed_buf.reshape(batch_size, n_actions), dtype=torch.float32)
        flat_adv = torch.as_tensor(advantage_np.reshape(batch_size), dtype=torch.float32)
        flat_ret = torch.as_tensor(returns_np.reshape(batch_size), dtype=torch.float32)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_logp = torch.as_tensor(logp_buf.reshape(batch_size), dtype=torch.float32)

        a_losses, c_losses = [], []
        for _ in range(K):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch):
                mb = indices[start:start + minibatch]
                mb_obs = flat_obs[mb]
                mb_unsquashed = flat_unsquashed[mb]
                mb_adv = flat_adv[mb]
                mb_ret = flat_ret[mb]
                mb_logp_old = flat_logp[mb]

                new_logp, entropy = model.evaluate(mb_obs, mb_unsquashed)
                new_values = critic(mb_obs)

                ratio = (new_logp - mb_logp_old).exp()
                unclamped_obj = ratio * mb_adv
                clamped_obj = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * mb_adv
                actor_loss = -torch.min(unclamped_obj, clamped_obj).mean()
                actor_loss -= entropy_coef * entropy.mean()
                critic_loss = F.mse_loss(new_values, mb_ret)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                a_losses.append(actor_loss.item())
                c_losses.append(critic_loss.item())

        return obs, float(np.mean(a_losses)), float(np.mean(c_losses)), rew_buf.sum(0).mean()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "policy.pt")

    actor_losses, critic_losses = [], []
    for e in range(epochs):
        lr_now = lr_schedule(e)
        for g in actor_optimizer.param_groups: g['lr'] = lr_now
        for g in critic_optimizer.param_groups: g['lr'] = lr_now
        obs, a_loss, c_loss, mean_rew = run_epoch(obs, entropy_schedule(e))
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)
        if e % 250 == 0:
            recent_ret = float(np.mean(ep_returns_log[-50:])) if ep_returns_log else float("nan")
            print(f"epoch {e:6d} | actor {a_loss:+.3f} | critic {c_loss:.3f} | window rew {mean_rew:+.3f} | recent ep return {recent_ret:+.3f}")
            if e > 0:
                torch.save(model.state_dict(), ckpt_path)
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
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--render", action="store_true", help="show a live demo episode every 1000 batches")

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--checkpoint", required=True)
    p_demo.add_argument("--episodes", type=int, default=10)

    p_eval = sub.add_parser("eval", help="headless evaluation over many episodes")
    p_eval.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "policy.pt"))
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--threshold", type=float, default=150.0)
    p_eval.add_argument("--stochastic", action="store_true", help="sample actions instead of using the distribution mean")

    p_rec = sub.add_parser("record", help="save mp4/gif demo videos of the best evaluation episodes")
    p_rec.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "policy.pt"))
    p_rec.add_argument("--episodes", type=int, default=20, help="episodes to roll out as candidates")
    p_rec.add_argument("--keep", type=int, default=5, help="how many of the best episodes to save")
    p_rec.add_argument("--seed", type=int, default=0)
    p_rec.add_argument("--fps", type=int, default=50)
    p_rec.add_argument("--gif", action="store_true", help="also write a gif next to each mp4")
    p_rec.add_argument("--min-return", type=float, default=None, help="only save episodes at or above this return")
    p_rec.add_argument("--prefix", default="lunarlander")
    p_rec.add_argument("--stochastic", action="store_true", help="sample actions instead of using the distribution mean")

    args = parser.parse_args()
    if args.cmd == "train":
        train(render=args.render)
    elif args.cmd == "demo":
        demo(checkpoint=args.checkpoint, episodes=args.episodes)
    elif args.cmd == "record":
        record(checkpoint=args.checkpoint,
               episodes=args.episodes,
               keep=args.keep,
               seed=args.seed,
               deterministic=not args.stochastic,
               fps=args.fps,
               gif=args.gif,
               min_return=args.min_return,
               prefix=args.prefix)
    elif args.cmd == "eval":
        evaluate_policy(checkpoint=args.checkpoint,
                        episodes=args.episodes,
                        seed=args.seed,
                        deterministic=not args.stochastic,
                        threshold=args.threshold)
