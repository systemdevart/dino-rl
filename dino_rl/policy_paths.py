"""Shared checkpoint locations for trained Dino policies."""

import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DQN_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dino_runner.pth")
PPO_BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dino_ppo_best.pth")
PPO_LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dino_ppo_last.pth")


def get_ppo_checkpoint_paths(
    env_backend: str = "sim",
    observation_mode: str = "feature",
) -> tuple[str, str]:
    """Return (best, last) PPO checkpoint paths for the given backend."""
    if env_backend == "sim" and observation_mode == "feature":
        return PPO_BEST_CHECKPOINT_PATH, PPO_LAST_CHECKPOINT_PATH

    suffix = (
        env_backend
        if observation_mode == "feature"
        else f"{env_backend}_{observation_mode}"
    )
    stem = os.path.join(CHECKPOINT_DIR, f"dino_ppo_{suffix}")
    return f"{stem}_best.pth", f"{stem}_last.pth"


def default_policy_path() -> str:
    """Prefer the best PPO checkpoint when available, otherwise fall back to DQN."""
    if os.path.isfile(PPO_BEST_CHECKPOINT_PATH):
        return PPO_BEST_CHECKPOINT_PATH
    return DQN_CHECKPOINT_PATH
