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

## Algorithms and Results

All algorithms trained on the same Dino environment with 8-dimensional feature vectors (obstacle distance, size, speed, dino state). Reward: +0.01 per step, -10.0 on crash.

### Converged

| Algorithm | Eval Avg | Eval Max | Sutton & Barto Reference |
|---|---|---|---|
| One-step Actor-Critic | 19,573 | 30,050 | Ch 13.5 |
| VanillaDQN | 1,517 | 1,984 | Mnih 2015 |
| DuelingDoubleDQN | 1,231 | 1,996 | Wang 2016 |
| DuelingDoubleDQN + PER | 1,053 | 1,771 | Schaul 2016 |
| DoubleDQN | 342 | 568 | van Hasselt 2016 |

### Did Not Converge (educational value in understanding *why*)

| Algorithm | Train Avg | Eval Avg | Limitation |
|---|---|---|---|
| REINFORCE | 103 | 62 | MC returns insensitive to episode quality with gamma=0.99 |
| REINFORCE + baseline | 95 | 61 | Baseline doesn't fix the MC return problem |
| PPO | 97 | 61 | Clipping slows learning; needs 10x more updates |
| N-step Actor-Critic | 87 | 60 | Fewer updates/episode than 1-step AC |
| A2C | 76 | 60 | Briefly peaked at 19,616 then collapsed |

### Key Educational Takeaways

1. **DQN methods converge stably** via experience replay + target networks
2. **One-step Actor-Critic is the only policy gradient method that converges** -- per-step TD updates propagate the crash penalty backward through the value function
3. **REINFORCE can't solve this task** -- with gamma=0.99, the discounted return G_t converges to ~1.0 regardless of episode length, providing no learning signal
4. **Actor-Critic instability motivates PPO** -- AC oscillated between eval 60 and 24,000; PPO's clipping prevents this but also prevents rapid learning
5. **A2C's brief spike then collapse** demonstrates why trust regions (PPO) or replay buffers (DQN) matter for stable learning

## Environment

The `dino_rl/env.py` module contains a pure-Python Chrome Dinosaur game simulation with:
- Accurate physics (acceleration from speed 6 to 13)
- Cacti and pterodactyl obstacles
- Domain randomization (obstacle clusters, variable dino position)
- 8-dimensional feature extraction for RL agents

## Requirements

- Python >= 3.8
- PyTorch
- NumPy, Matplotlib
- TensorBoard (optional, for training visualization)
