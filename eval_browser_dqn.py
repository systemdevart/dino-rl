"""Evaluate a sim-trained feature-DQN in the REAL browser (deterministic env).

Usage:
    python eval_browser_dqn.py [checkpoint] [episodes] [step_cap]

Defaults reproduce the parity result: checkpoints/dino_runner_feat_a.pth (the
sim-trained DQN, ~5.9k in the sim) scores the step-cap in every episode --
score 6500 at cap 22000, ~18.9k at cap 60000 -- it simply never crashes.

Runs at frames_per_action=1 (decide every frame): the game pauses between
frames via the deterministic env, so inference latency doesn't desync actions.
"""
import sys

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from dino_rl.browser_env import ChromeDinoGame
from dino_rl.feature_contract import FEATURE_DIM
from dino_rl.networks import DuelingDQN
from dino_rl.train_dqn import greedy_feature_actions

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dino_runner_feat_a.pth"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
STEP_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 22000

model = DuelingDQN(FEATURE_DIM, 3)
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
model.eval()
NO_DUCK = bool(ck.get("no_duck", False)) if isinstance(ck, dict) else False


@torch.no_grad()
def act(features) -> int:
    states = torch.tensor(np.array([features]), dtype=torch.float32)
    q_values = model(states)
    return int(greedy_feature_actions(
        q_values,
        no_duck=NO_DUCK,
    ).item())


def episode(_i) -> int:
    game = ChromeDinoGame(headless=True)
    game.start()
    game.enable_deterministic()
    game.restart_deterministic()
    state = game.env_step(0, 1)
    steps = 0
    while steps < STEP_CAP and state is not None and not state["crashed"]:
        state = game.env_step(act(state["features"]), 1)
        steps += 1
    game.close()
    return int(state["score"]) if state else 0


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=min(EPISODES, 8)) as ex:
        scores = np.array(list(ex.map(episode, range(EPISODES))))
    print(f"{CKPT} | browser fpa=1 | n={len(scores)} cap={STEP_CAP}: "
          f"mean={scores.mean():.0f} min={scores.min()} max={scores.max()} "
          f">=5000: {int(np.sum(scores >= 5000))}/{len(scores)}")
    print("scores:", sorted(scores.tolist()))
