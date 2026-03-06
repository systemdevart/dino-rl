"""
Shared infrastructure for RL algorithm implementations.

Provides:
- DinoFeatureEnv: wrapper that returns feature vectors instead of images
- evaluate(): deterministic policy evaluation
- plot_training(): save training curve plots
- save_results(): persist JSON results for comparison
- create_writer(): create TensorBoard SummaryWriter for an algorithm
"""
import json
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dino_rl.env import DinoRunEnv

FEATURE_DIM = 8   # Number of features from env.get_features()
ACTION_SIZE = 2    # 0: do nothing, 1: jump
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


class DinoFeatureEnv:
    """
    Wrapper around DinoRunEnv that returns feature vectors as state.

    Usage:
        env = DinoFeatureEnv()
        state = env.reset()           # np.array of shape (8,)
        state, reward, done, info = env.step(action)
    """

    def __init__(self, domain_randomization=False, feature_noise=0.0):
        self.env = DinoRunEnv(
            domain_randomization=domain_randomization,
            feature_noise=feature_noise,
        )
        self.action_space = self.env.action_space

    def reset(self):
        """Reset environment, return feature state."""
        self.env.reset()
        return self.env.get_features()

    def step(self, action):
        """
        Take a step, return (state, reward, done, info).

        info contains 'score' (the game score).

        Reward shaping:
        - Per-step survival reward: +0.01 (reduced from the env's default +0.1).
          With gamma=0.99, the discounted sum of +0.01/step converges to ~1.0,
          which is much smaller than the crash penalty of -10.  This ensures that
          the crash signal dominates the return for short (bad) episodes:
              60-step episode:  G_0 ≈ -5.0  (crash dominates → bad)
              300-step episode: G_0 ≈  0.0  (break-even)
              600-step episode: G_0 ≈ +1.0  (survival wins → good)
          Without this scaling, G_0 ≈ +10 for ALL episode lengths, making
          REINFORCE-family methods unable to distinguish good from bad episodes.
        - Crash penalty: -10.0 (amplified from the env's default -1.0).
        """
        _, reward, score, done = self.env.step(action)
        state = self.env.get_features()
        if done:
            reward = -10.0
        else:
            reward = 0.01
        return state, reward, done, {'score': score}

    def get_score(self):
        return self.env.get_score()


def evaluate(policy_fn, n_episodes=20, max_steps=25000):
    """
    Evaluate a deterministic policy over n_episodes.

    Args:
        policy_fn: callable(state: np.ndarray) -> int action
        n_episodes: number of episodes to run
        max_steps: max steps per episode

    Returns:
        dict with 'avg', 'min', 'max', 'scores' keys
    """
    env = DinoFeatureEnv(domain_randomization=False)
    scores = []

    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy_fn(state)
            state, reward, done, info = env.step(action)
            if done:
                break
        scores.append(info['score'])

    return {
        'avg': float(np.mean(scores)),
        'min': int(np.min(scores)),
        'max': int(np.max(scores)),
        'scores': scores,
    }


def plot_training(scores, title, path, eval_scores=None):
    """
    Save a training curve plot.

    Args:
        scores: list of per-episode scores
        title: plot title
        path: file path to save the figure
        eval_scores: optional list of (episode, score) tuples for eval overlay
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plot_training] matplotlib not available, skipping plot: {path}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scores, alpha=0.3, label='Episode score')

    # Rolling average
    if len(scores) >= 50:
        window = min(100, len(scores) // 5)
        rolling = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(scores)), rolling, label=f'{window}-ep avg')

    if eval_scores:
        eps, vals = zip(*eval_scores)
        ax.plot(eps, vals, 'r-o', markersize=3, label='Eval avg')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training plot: {path}")


def save_results(algo_name, train_scores, eval_result=None):
    """
    Save training and eval results to JSON.

    Args:
        algo_name: algorithm identifier (e.g., 'reinforce')
        train_scores: list of per-episode training scores
        eval_result: optional dict from evaluate()
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'algorithm': algo_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_episodes': len(train_scores),
        'train_scores': train_scores,
        'train_avg': float(np.mean(train_scores[-100:])) if train_scores else 0,
    }
    if eval_result:
        result['eval'] = eval_result

    path = os.path.join(RESULTS_DIR, f'{algo_name}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results: {path}")


LOGS_DIR = os.path.join(RESULTS_DIR, 'runs')


def create_writer(algo_name):
    """Create a TensorBoard SummaryWriter for the given algorithm."""
    log_dir = os.path.join(LOGS_DIR, algo_name)
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
