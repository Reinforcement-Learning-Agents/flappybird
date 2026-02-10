import os
import csv
import time
import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import flappy_bird_gymnasium  # noqa: F401
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    env_id: str = "FlappyBird-v0"
    seed: int = 0
    device: str = "cpu"

    gamma: float = 0.99
    lr: float = 1e-4

    buffer_size: int = 200_000
    batch_size: int = 256

    learn_start: int = 10_000
    train_every: int = 4
    target_update_every: int = 2_000

    max_steps: int = 1_000_000  # totale step ambiente
    max_episode_steps: int | None = None  

    #LINEAR EPSILON DECAY
    #eps_start: float = 1.0
    #eps_end: float = 0.05
    #eps_decay_steps: int = 150_000  # decay lineare su questi step

    # EXPONENTIAL EPSILON DECAY
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 700_000  # decay esponenziale su questi step

    grad_clip: float = 10.0

    # reward shaping (opzionale)
    survival_reward: float = 0.05

    # eval / logging
    eval_every_steps: int = 25_000
    eval_episodes: int = 30

    # output
    run_dir: str = "results/ddqn"
    save_video_on_best: bool = True
    video_eval_episodes: int = 5  # quanti episodi registrare quando fai best


CFG = Config()
CFG.run_dir = f"results/ddqn_seed{CFG.seed}_1Msteps"


# -----------------------------
# Utils
# -----------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(base: str):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base, "videos"), exist_ok=True)


def obs_to_float(obs) -> np.ndarray:
    # Stato numerico: lo rendiamo 1D float32
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    # un clip leggero evita exploding valori rari
    return np.clip(x, -1e3, 1e3)


def make_env(render_mode=None):
    env = gym.make(CFG.env_id, render_mode=render_mode)
    env.reset(seed=CFG.seed)
    env.action_space.seed(CFG.seed)
    return env


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)

    def add(self, s, a, r, s2, done: bool):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )


# -----------------------------
# Q Network
# -----------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def greedy_action(qnet: nn.Module, s: np.ndarray) -> int:
    x = torch.as_tensor(s, dtype=torch.float32, device=CFG.device).unsqueeze(0)
    q = qnet(x).squeeze(0)
    return int(torch.argmax(q).item())


# --- epsilon schedule (logspace) ---
_eps_arr = None

def epsilon_by_step(step: int) -> float:
    global _eps_arr
    if _eps_arr is None:
        # genera una volta sola
        _eps_arr = np.logspace(start=0, stop=-2, num=CFG.eps_decay_steps, base=10).astype(np.float32)
        _eps_arr = np.clip(_eps_arr, CFG.eps_end, CFG.eps_start)

    if step < CFG.eps_decay_steps:
        return float(_eps_arr[step])
    return float(CFG.eps_end)



def eps_greedy_action(qnet: nn.Module, env: gym.Env, s: np.ndarray, eps: float) -> int:
    if random.random() < eps:
        return int(env.action_space.sample())
    return greedy_action(qnet, s)


# -----------------------------
# Training step (Double DQN)
# -----------------------------
def train_step(qnet, qtarget, optim, rb: ReplayBuffer):
    s, a, r, s2, d = rb.sample(CFG.batch_size)

    s_t = torch.as_tensor(s, dtype=torch.float32, device=CFG.device)
    a_t = torch.as_tensor(a, dtype=torch.int64, device=CFG.device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=CFG.device)
    s2_t = torch.as_tensor(s2, dtype=torch.float32, device=CFG.device)
    d_t = torch.as_tensor(d, dtype=torch.float32, device=CFG.device)

    # Q(s,a)
    q_sa = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN:
        # online selects best action in s2
        a2 = qnet(s2_t).argmax(dim=1)
        # target evaluates it
        q_next = qtarget(s2_t).gather(1, a2.unsqueeze(1)).squeeze(1)
        y = r_t + CFG.gamma * q_next * (1.0 - d_t)

    # MSE loss
    loss = F.mse_loss(q_sa, y)

    # Huber loss
    #loss = F.smooth_l1_loss(q_sa, y)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), CFG.grad_clip)
    optim.step()

    return float(loss.item())


# -----------------------------
# Evaluation (greedy)
# -----------------------------
@torch.no_grad()
def evaluate(qnet: nn.Module, episodes: int) -> float:
    env = make_env()
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = obs_to_float(obs)
        done = False
        ep_ret = 0.0
        steps = 0

        while not done:
            a = greedy_action(qnet, s)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            if not done:
                r += CFG.survival_reward

            ep_ret += float(r)
            s = obs_to_float(obs2)

            steps += 1
            if CFG.max_episode_steps is not None and steps >= CFG.max_episode_steps:
                break

        rets.append(ep_ret)

    env.close()
    return float(np.mean(rets))


@torch.no_grad()
def record_videos(qnet: nn.Module, episodes: int, out_folder: str, prefix: str):
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=out_folder,
        episode_trigger=lambda ep: True,
        name_prefix=prefix,
    )

    for _ in range(episodes):
        obs, _ = env.reset()
        s = obs_to_float(obs)
        done = False
        steps = 0

        while not done:
            a = greedy_action(qnet, s)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = obs_to_float(obs2)

            steps += 1
            if CFG.max_episode_steps is not None and steps >= CFG.max_episode_steps:
                break

    env.close()


# -----------------------------
# Plot + CSV logging
# -----------------------------
def save_plot(eval_steps, eval_scores, out_png):
    plt.figure()
    plt.plot(eval_steps, eval_scores)
    plt.xlabel("Environment steps")
    plt.ylabel("Eval avg return")
    plt.title("DDQN FlappyBird (Double DQN + MSE_loss)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def append_csv(csv_path, row, header=None):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if (not exists) and header is not None:
            w.writerow(header)
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    set_seeds(CFG.seed)
    ensure_dirs(CFG.run_dir)

    env = make_env()
    obs, _ = env.reset()
    s0 = obs_to_float(obs)
    state_dim = int(s0.shape[0])
    n_actions = int(env.action_space.n)

    qnet = QNet(state_dim, n_actions).to(CFG.device)
    qtarget = QNet(state_dim, n_actions).to(CFG.device)
    qtarget.load_state_dict(qnet.state_dict())
    qtarget.eval()

    #SGD Optimizer + momentum
    optim = torch.optim.SGD(qnet.parameters(), lr=CFG.lr, momentum=0.9)

    #Optimizer ADAM
    #optim = torch.optim.Adam(qnet.parameters(), lr=CFG.lr)

    rb = ReplayBuffer(CFG.buffer_size, state_dim)

    # logging
    metrics_csv = os.path.join(CFG.run_dir, "metrics.csv")
    plot_path = os.path.join(CFG.run_dir, "plots", "eval_curve.png")

    eval_steps = []
    eval_scores = []
    best_eval = -1e9
    best_step = 0

    # training loop
    obs, _ = env.reset()
    s = obs_to_float(obs)
    ep_return = 0.0
    ep_len = 0
    episode = 1

    losses_window = deque(maxlen=200)
    t0 = time.time()

    for step in range(1, CFG.max_steps + 1):
        eps = epsilon_by_step(step)
        a = eps_greedy_action(qnet, env, s, eps)

        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        if not done:
            r += CFG.survival_reward

        s2 = obs_to_float(obs2)

        rb.add(s, a, float(r), s2, done)

        s = s2
        ep_return += float(r)
        ep_len += 1

        # episode reset
        if done or (CFG.max_episode_steps is not None and ep_len >= CFG.max_episode_steps):
            append_csv(
                metrics_csv,
                row=[step, episode, "train_episode", ep_return, ep_len, eps, "", ""],
                header=["step", "episode", "type", "value", "ep_len", "eps", "loss_avg200", "best_eval"],
            )
            obs, _ = env.reset()
            s = obs_to_float(obs)
            ep_return = 0.0
            ep_len = 0
            episode += 1

        # train
        if step >= CFG.learn_start and step % CFG.train_every == 0 and rb.size >= CFG.batch_size:
            loss = train_step(qnet, qtarget, optim, rb)
            losses_window.append(loss)

        # target update
        if step >= CFG.learn_start and step % CFG.target_update_every == 0:
            qtarget.load_state_dict(qnet.state_dict())

        # eval
        if step % CFG.eval_every_steps == 0:
            avg_eval = evaluate(qnet, CFG.eval_episodes)
            eval_steps.append(step)
            eval_scores.append(avg_eval)

            loss_avg = float(np.mean(losses_window)) if len(losses_window) else float("nan")

            # save latest
            torch.save(qnet.state_dict(), os.path.join(CFG.run_dir, "dqn_flappy_latest.pt"))

            # save plot + arrays
            np.save(os.path.join(CFG.run_dir, "eval_steps.npy"), np.array(eval_steps, dtype=np.int64))
            np.save(os.path.join(CFG.run_dir, "eval_scores.npy"), np.array(eval_scores, dtype=np.float32))
            save_plot(eval_steps, eval_scores, plot_path)

            # best checkpoint + optional video
            improved = avg_eval > best_eval
            if improved:
                best_eval = avg_eval
                best_step = step
                torch.save(qnet.state_dict(), os.path.join(CFG.run_dir, "dqn_flappy_best.pt"))

                if CFG.save_video_on_best:
                    prefix = f"dqn_best_step{step}_ret{avg_eval:.2f}"
                    record_videos(
                        qnet,
                        episodes=CFG.video_eval_episodes,
                        out_folder=os.path.join(CFG.run_dir, "videos"),
                        prefix=prefix,
                    )

            # print + csv
            elapsed = time.time() - t0
            print(
                f"[step {step:7d}] eval_avg={avg_eval:8.3f} | best={best_eval:8.3f} (step {best_step}) "
                f"| eps={eps:5.3f} | loss_avg200={loss_avg:7.4f} | elapsed={elapsed/60:.1f}m"
            )

            append_csv(
                metrics_csv,
                row=[step, episode, "eval", avg_eval, "", eps, loss_avg, best_eval],
            )

    env.close()
    print("\n Finito.")
    print(f" - Best model:   {os.path.join(CFG.run_dir, 'dqn_flappy_best.pt')}")
    print(f" - Latest model: {os.path.join(CFG.run_dir, 'dqn_flappy_latest.pt')}")
    print(f" - Plot:         {plot_path}")
    print(f" - Videos:       {os.path.join(CFG.run_dir, 'videos')}")


if __name__ == "__main__":
    main()
