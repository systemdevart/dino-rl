# Reinforcement Learning on Chrome Dino Game

A comprehensive educational implementation of RL algorithms from Sutton & Barto, trained on a pure-Python Chrome Dinosaur game simulation. Each algorithm file contains detailed comments explaining the theory, math, and design decisions.

## Quick Start

```bash
pip install -e .

# Train all algorithms with TensorBoard logging
python -m dino_rl.algorithms.train_all

# View training curves
tensorboard --logdir results/runs

# Train the standalone DQN agent
python -m dino_rl.train_dqn

# Play in the browser with a trained model
python -m dino_rl.play_browser
```

## Project Structure

```
Reinforcement_Learning/
├── pyproject.toml               # Package metadata and dependencies
├── requirements.txt             # pip install -r convenience
├── results/                     # JSON results + training plots
├── checkpoints/                 # Saved model weights
│   └── dino_runner.pth
└── dino_rl/                     # Main package
    ├── __init__.py
    ├── env.py                   # Pure Python Chrome Dino game simulation
    ├── common.py                # Shared env wrapper, eval, plotting, TB logging
    ├── networks.py              # Shared DuelingDQN architecture
    ├── train_dqn.py             # Standalone DQN training script
    ├── play_browser.py          # Play trained model in real Chrome browser
    └── algorithms/
        ├── train_all.py         # Train all algorithms sequentially
        ├── value_network.py     # Semi-gradient TD(0) value network
        ├── reinforce.py         # REINFORCE (Monte Carlo policy gradient)
        ├── reinforce_baseline.py# REINFORCE with learned baseline
        ├── actor_critic.py      # One-step Actor-Critic
        ├── actor_critic_nstep.py# N-step Actor-Critic
        ├── td_lambda.py         # True Online TD(lambda)
        ├── dqn_progression.py   # DQN -> Double -> Dueling -> PER
        ├── ppo.py               # Proximal Policy Optimization
        └── a2c.py               # Advantage Actor-Critic (synchronous)
```

## Standalone DQN (Browser Play)

The standalone Dueling Double DQN (`dino_rl/train_dqn.py`) is the model used for browser play. Trained on the 1:1 faithful environment:

| Metric              | Value |
| ------------------- | ----- |
| Eval Avg (30 games) | 1,562 |
| Eval Median         | 1,564 |
| Eval Max            | 2,169 |
| Eval Min            | 635   |

## Algorithm Comparison

All algorithms trained on the same Dino environment with 8-dimensional feature vectors (obstacle distance, size, speed, dino state). Reward: +0.01 per step, -10.0 on crash. Scores use the real game's formula: `round(distance * 0.025)`.

### Key Educational Takeaways

1. **DQN methods converge stably** via experience replay + target networks
2. **Reward shaping matters** -- +0.1/step survival reward swamps the -10 crash penalty; reducing to +0.01/step makes the crash signal dominate for short (bad) episodes
3. **REINFORCE can't solve this task** -- with gamma=0.99, the discounted return G_t converges to ~1.0 regardless of episode length, providing no learning signal
4. **Actor-Critic instability** -- oscillates between good and bad policies without replay buffers for stabilisation

## Environment

The `dino_rl/env.py` module is a **1:1 faithful port** of Chromium's T-Rex Runner (`offline.js`):

- Exact physics: speed-dependent jump velocity (`-10 - speed/10`), gravity 0.6, endJump() at MAX_JUMP_HEIGHT
- Exact obstacle types: CACTUS_SMALL (17x35), CACTUS_LARGE (25x50), PTERODACTYL (46x40, minSpeed 8.5)
- Exact gap formula: `round(width * speed + minGap * 0.6)`, maxGap = 1.5x
- Two-phase collision: broad-phase 1px-margin boxes + narrow-phase per-part AABB
- Exact scoring: `round(distanceRan * 0.025)`
- Multi-size obstacles (1-3 cacti based on speed), duplicate type prevention
- 8-dimensional feature extraction for RL agents

## Study Plan

See **[STUDY_PLAN.md](STUDY_PLAN.md)** for a detailed learning guide: what to read, what to research, commands to run, metrics to track, and a 6-week study schedule covering all algorithms from foundations to advanced methods.

## Requirements

- Python >= 3.8
- PyTorch
- NumPy, Matplotlib
- TensorBoard (optional, for training visualization)
