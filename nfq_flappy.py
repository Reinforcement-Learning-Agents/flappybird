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


# =========================
# CONFIG 
# =========================
@dataclass
class Config:
    env_id: str = "FlappyBird-v0"
    seed: int = 2
    device: str = "cpu"

    gamma: float = 0.99
    lr: float = 1e-3  # NFQ spesso usa lr più alta del DQN

    # budget in env steps (confrontabile con DQN/DDQN)
    max_steps: int = 600_000

    # NFQ: fitted Q-iteration on batches
    batch_size: int = 5_000          # transizioni per fitted-iteration
    epochs_per_batch: int = 25       # epoche supervisionate sul batch
    minibatch: int = 256
    grad_clip: float = 10.0

    #Epsilon fisso per data collection
    epsilon: float = 0.5

    # reward shaping (mettere lo stesso del DQN/DDQN!)
    survival_reward: float = 0.05

    # eval/log 
    eval_every_steps: int = 25_000
    eval_episodes: int = 30

    # output
    run_dir: str = "results/nfq"
    save_video_on_best: bool = True
    video_eval_episodes: int = 2


CFG = Config()
CFG.run_dir = f"results/nfq_seed{CFG.seed}"

# =========================
# UTILS
# =========================
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(base: str):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "plots"), exist_ok=True)
    os.makedirs(os.path.join(base, "videos"), exist_ok=True)


def obs_to_float(obs) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.clip(x, -1e3, 1e3)


def make_env(render_mode=None):
    env = gym.make(CFG.env_id, render_mode=render_mode)
    env.reset(seed=CFG.seed)
    env.action_space.seed(CFG.seed)
    return env


# =========================
# MODEL
# =========================
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


def epsilon_greedy_action_prof(qnet: nn.Module, env: gym.Env, s: np.ndarray) -> int:
    """
    PROFF STYLE:
      epsilon fisso (exploration vs exploitation)
      random_pi(state) vs nfq_pi(state)
    """
    if random.random() < CFG.epsilon:
        return int(env.action_space.sample())
    return greedy_action(qnet, s)


# =========================
# EVAL + VIDEO 
# =========================
@torch.no_grad()
def evaluate(qnet: nn.Module, episodes: int) -> float:
    env = make_env()
    qnet.eval()
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = obs_to_float(obs)
        done = False
        ep_ret = 0.0

        while not done:
            a = greedy_action(qnet, s)  # greedy puro in eval
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if not done:
                r += CFG.survival_reward
            ep_ret += float(r)
            s = obs_to_float(obs2)

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
    qnet.eval()

    for _ in range(episodes):
        obs, _ = env.reset()
        s = obs_to_float(obs)
        done = False
        while not done:
            a = greedy_action(qnet, s)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = obs_to_float(obs2)

    env.close()


def save_plot(eval_steps, eval_scores, out_png):
    plt.figure()
    plt.plot(eval_steps, eval_scores)
    plt.xlabel("Environment steps")
    plt.ylabel("Score")
    plt.title("NFQ FlappyBird (prof-style epsilon, fitted Q-iteration)")
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


# =========================
# NFQ core: fitted Q-iteration (target congelata)
# =========================
def fitted_iteration(qnet: nn.Module, qtarget: nn.Module, optim, batch, losses_window: deque):
    """
    NFQ / Fitted Q Iteration:
      y = r + gamma * max_a Q_target(s',a) * (1-done)
    IMPORTANT:
      qtarget è congelata durante tutta l'iterazione.
    """
    s, a, r, s2, d = batch

    s_t  = torch.as_tensor(s,  dtype=torch.float32, device=CFG.device)
    a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=CFG.device)
    r_t  = torch.as_tensor(r,  dtype=torch.float32, device=CFG.device)
    s2_t = torch.as_tensor(s2, dtype=torch.float32, device=CFG.device)
    d_t  = torch.as_tensor(d,  dtype=torch.float32, device=CFG.device)

    N = s_t.shape[0]
    idx = np.arange(N)

    qnet.train()
    qtarget.eval()

    for _ in range(CFG.epochs_per_batch):
        np.random.shuffle(idx)
        for start in range(0, N, CFG.minibatch):
            mb = idx[start:start + CFG.minibatch]

            q_sa = qnet(s_t[mb]).gather(1, a_t[mb].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = qtarget(s2_t[mb]).max(dim=1).values
                y = r_t[mb] + CFG.gamma * q_next * (1.0 - d_t[mb])

            # MSE LOSS
            loss = F.mse_loss(q_sa, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), CFG.grad_clip)
            optim.step()

            losses_window.append(float(loss.item()))

    qnet.eval()


# =========================
# MAIN
# =========================
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

    # SGD optimizer + momentum
    # optim = torch.optim.SGD(qnet.parameters(), lr=CFG.lr, momentum=0.9)

    # ADAM optimizer
    optim = torch.optim.Adam(qnet.parameters(), lr=CFG.lr)

    # batch buffer NFQ (si riempie e poi fai fitted iteration)
    s_buf  = np.zeros((CFG.batch_size, state_dim), dtype=np.float32)
    a_buf  = np.zeros((CFG.batch_size,), dtype=np.int64)
    r_buf  = np.zeros((CFG.batch_size,), dtype=np.float32)
    s2_buf = np.zeros((CFG.batch_size, state_dim), dtype=np.float32)
    d_buf  = np.zeros((CFG.batch_size,), dtype=np.float32)
    buf_i = 0

    # logging 
    metrics_csv = os.path.join(CFG.run_dir, "metrics.csv")
    plot_path = os.path.join(CFG.run_dir, "plots", "eval_curve.png")

    eval_steps = []
    eval_scores = []

    best_eval = -1e9
    best_step = 0

    losses_window = deque(maxlen=200)

    # episode bookkeeping (solo per logging)
    obs, _ = env.reset()
    s = obs_to_float(obs)
    ep_ret = 0.0
    ep_len = 0
    episode = 1

    t0 = time.time()

    append_csv(
        metrics_csv,
        row=[],
        header=["step", "episode", "type", "value", "ep_len", "eps", "loss_avg200", "best_eval"],
    )

    for step in range(1, CFG.max_steps + 1):
        #epsilon fisso
        a = epsilon_greedy_action_prof(qnet, env, s)

        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        if not done:
            r += CFG.survival_reward

        s2 = obs_to_float(obs2)

        # store transition into current batch
        s_buf[buf_i] = s
        a_buf[buf_i] = a
        r_buf[buf_i] = float(r)
        s2_buf[buf_i] = s2
        d_buf[buf_i] = 1.0 if done else 0.0
        buf_i += 1

        # when batch full -> snapshot target and fitted iteration
        if buf_i == CFG.batch_size:
            qtarget.load_state_dict(qnet.state_dict())  # freeze target for this iteration
            batch = (s_buf, a_buf, r_buf, s2_buf, d_buf)
            fitted_iteration(qnet, qtarget, optim, batch, losses_window)
            buf_i = 0  # reset

        # advance
        s = s2
        ep_ret += float(r)
        ep_len += 1

        if done:
            loss_avg = float(np.mean(losses_window)) if len(losses_window) else float("nan")
            append_csv(
                metrics_csv,
                row=[step, episode, "train_episode", ep_ret, ep_len, CFG.epsilon, loss_avg, best_eval],
            )
            obs, _ = env.reset()
            s = obs_to_float(obs)
            ep_ret = 0.0
            ep_len = 0
            episode += 1

        # eval / save (stesso protocollo DQN)
        if step % CFG.eval_every_steps == 0:
            avg_eval = evaluate(qnet, CFG.eval_episodes)
            eval_steps.append(step)
            eval_scores.append(avg_eval)

            # save latest
            torch.save(qnet.state_dict(), os.path.join(CFG.run_dir, "nfq_latest.pt"))

            # arrays + plot
            np.save(os.path.join(CFG.run_dir, "eval_steps.npy"), np.array(eval_steps, dtype=np.int64))
            np.save(os.path.join(CFG.run_dir, "eval_scores.npy"), np.array(eval_scores, dtype=np.float32))
            save_plot(eval_steps, eval_scores, plot_path)

            improved = avg_eval > best_eval
            if improved:
                best_eval = avg_eval
                best_step = step
                torch.save(qnet.state_dict(), os.path.join(CFG.run_dir, "nfq_best.pt"))

                if CFG.save_video_on_best:
                    prefix = f"nfq_best_step{step}_ret{avg_eval:.2f}"
                    record_videos(
                        qnet,
                        episodes=CFG.video_eval_episodes,
                        out_folder=os.path.join(CFG.run_dir, "videos"),
                        prefix=prefix,
                    )

            loss_avg = float(np.mean(losses_window)) if len(losses_window) else float("nan")
            elapsed = time.time() - t0
            print(
                f"[NFQ step {step:7d}] eval_avg={avg_eval:7.3f} | best={best_eval:7.3f} (step {best_step}) "
                f"| eps={CFG.epsilon:5.3f} | loss_avg200={loss_avg:7.4f} | elapsed={elapsed/60:.1f}m"
            )

            append_csv(
                metrics_csv,
                row=[step, episode, "eval", avg_eval, "", CFG.epsilon, loss_avg, best_eval],
            )

    env.close()
    torch.save(qnet.state_dict(), os.path.join(CFG.run_dir, "nfq_final.pt"))
    print("\n Finito NFQ.")
    print(f" - Best model:   {os.path.join(CFG.run_dir, 'nfq_best.pt')}")
    print(f" - Latest model: {os.path.join(CFG.run_dir, 'nfq_latest.pt')}")
    print(f" - Plot:         {plot_path}")
    print(f" - Videos:       {os.path.join(CFG.run_dir, 'videos')}")


if __name__ == "__main__":
    main()
