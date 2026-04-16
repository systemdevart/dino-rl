# Reinforcement Learning Study Plan — Chrome Dino Game

A hands-on guide to learning RL by building agents for the Chrome T-Rex Runner.
This project implements 12 algorithms from scratch and a zero-ML heuristic baseline,
all training on a pixel-perfect Python simulation of the original Chromium game.

---

## 1. Prerequisites

### Math You Need
- **Probability**: conditional probability, expectation, Bayes' rule
- **Linear algebra**: matrix multiplication, dot products (for neural network intuition)
- **Calculus**: chain rule, partial derivatives (for understanding backprop)
- **Statistics**: mean, variance, standard deviation, moving averages

### Python You Need
- NumPy array operations, broadcasting
- PyTorch basics: tensors, autograd, `nn.Module`, optimizers
- Matplotlib for plotting training curves

### Install & Verify
```bash
cd Reinforcement_Learning/
pip install -e ".[all]"

# Verify everything works
python -c "from dino_rl.env import DinoRunEnv; env = DinoRunEnv(); print('env OK')"
python -c "from dino_rl.networks import DuelingDQN; print('networks OK')"
python -c "from dino_rl.common import DinoFeatureEnv; print('common OK')"
```

---

## 2. Reading List (in order)

### Phase 1: Foundations
| What | Where | Why |
|---|---|---|
| Sutton & Barto Ch 1-3 | [incompleteideas.net/book](http://incompleteideas.net/book/the-book-2nd.html) | MDPs, returns, value functions — the language of RL |
| David Silver Lecture 1-3 | [YouTube playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) | Visual walkthrough of the same concepts |
| Our `dino_rl/env.py` | `python -c "from dino_rl.env import DinoRunEnv; help(DinoRunEnv)"` | Understand the environment you're training on |
| Chromium T-Rex source | Search "AcNielsen offline.js chromium" or read `components/neterror/resources/offline.js` | See the original JavaScript this env replicates |

### Phase 2: Value-Based Methods
| What | Where | Why |
|---|---|---|
| Sutton & Barto Ch 6 (TD Learning) | Same book | TD(0), bootstrapping, the core idea behind DQN |
| Sutton & Barto Ch 9.3 (Semi-gradient TD) | Same book | How to combine TD with function approximation |
| Our `dino_rl/algorithms/value_network.py` | Read the source | See semi-gradient TD(0) with a neural network |
| Sutton & Barto Ch 12 (Eligibility Traces) | Same book | TD(λ) — bridging TD(0) and Monte Carlo |
| Our `dino_rl/algorithms/td_lambda.py` | Read the source | Implementation of TD(λ) with eligibility traces |

### Phase 3: Policy Gradient Methods
| What | Where | Why |
|---|---|---|
| Sutton & Barto Ch 13 | Same book | REINFORCE, policy gradient theorem |
| [Policy Gradient Methods (Lilian Weng)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) | Blog post | Excellent visual explanation |
| Our `dino_rl/algorithms/reinforce.py` | Read the source | Vanilla REINFORCE |
| Our `dino_rl/algorithms/reinforce_baseline.py` | Read the source | Variance reduction with a baseline |
| Our `dino_rl/algorithms/actor_critic.py` | Read the source | One-step actor-critic (TD actor-critic) |
| Our `dino_rl/algorithms/actor_critic_nstep.py` | Read the source | N-step returns for lower bias |

### Phase 4: Advanced Methods
| What | Where | Why |
|---|---|---|
| [Playing Atari with Deep RL (Mnih 2013)](https://arxiv.org/abs/1312.5602) | arXiv | The original DQN paper |
| [Human-level control (Mnih 2015)](https://www.nature.com/articles/nature14236) | Nature | DQN with target networks and experience replay |
| [Double DQN (van Hasselt 2015)](https://arxiv.org/abs/1509.06461) | arXiv | Fixes overestimation bias |
| [Dueling DQN (Wang 2016)](https://arxiv.org/abs/1511.06581) | arXiv | Separate value and advantage streams |
| [Prioritized Experience Replay (Schaul 2016)](https://arxiv.org/abs/1511.05952) | arXiv | Learn more from surprising transitions |
| Our `dino_rl/algorithms/dqn_progression.py` | Read the source | All 4 DQN variants in one file |
| Our `dino_rl/train_dqn.py` | Read the source | Production DQN training (Dueling Double DQN) |
| [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347) | arXiv | Clipped surrogate objective, most popular method |
| Our `dino_rl/algorithms/ppo.py` | Read the source | PPO implementation |
| [A3C / A2C (Mnih 2016)](https://arxiv.org/abs/1602.01783) | arXiv | Advantage actor-critic |
| Our `dino_rl/algorithms/a2c.py` | Read the source | Synchronous advantage actor-critic |

### Phase 5: Beyond ML
| What | Where | Why |
|---|---|---|
| Our `dino_rl/heuristic_agent.py` | Read the source | When you don't need ML at all |
| Think about: state space analysis | See Section 6 below | Critical skill — know when RL is overkill |

---

## 3. Commands Reference

### Training Educational Algorithms (quick, ~2-5 min each)
```bash
# Train ALL algorithms sequentially (writes results/ JSONs + PNGs)
python -m dino_rl.algorithms.train_all

# Train a single algorithm
python -m dino_rl.algorithms.train_all --algo reinforce
python -m dino_rl.algorithms.train_all --algo actor_critic
python -m dino_rl.algorithms.train_all --algo ppo
python -m dino_rl.algorithms.train_all --algo dqn

# Available algorithms:
#   value_network, reinforce, reinforce_baseline, actor_critic,
#   actor_critic_nstep, td_lambda, a2c, ppo, dqn
```

### Training Production DQN (longer, ~30-60 min)
```bash
# Full DQN training with eval checkpointing
python -m dino_rl.train_dqn

# Saves best model to checkpoints/dino_runner.pth
```

### Evaluation
```bash
# Evaluate the heuristic agent (no training needed)
python -m dino_rl.heuristic_agent

# Evaluate the trained DQN
python -c "
from dino_rl.train_dqn import TRexRunner
runner = TRexRunner()
for i in range(10):
    score, steps, min_s = runner.run_eval_episode()
    print(f'Game {i+1}: avg={score}, min={min_s}, steps={steps}')
"
```

### Playing in Real Browser
```bash
# Launch Chrome with the trained DQN controlling the dino
python -m dino_rl.play_browser
# Requires: Chrome/Chromium installed, selenium
```

### TensorBoard (for educational algorithms)
```bash
tensorboard --logdir results/runs
# Then open http://localhost:6006
```

### PPO Autoresearch Notes (Apr 2026)

The best 30-minute PPO variant so far keeps the baseline hyperparameters and
adds a small score-progress reward bonus inside `dino_rl/algorithms/ppo.py`:

```python
score_delta = max(info['score'] - prev_score, 0)
reward = reward + 0.02 * score_delta
```

Two smaller cleanups shipped with it:
- `scheduler.step()` now runs after the PPO optimizer update
- rollout tensors use `torch.as_tensor(...)` instead of rebuilding tensors

This was the first PPO change that clearly beat the old baseline under the
same 30-minute wall-clock budget:

| PPO Run | Final Eval Avg | Final Eval Max | Notes |
|---|---|---|---|
| Baseline PPO | 1427.5 | 2208 | No score-progress shaping |
| Best PPO autoresearch run | 2140.1 | 5331 | `score_delta_coeff = 0.02` |

What to watch in TensorBoard for PPO:

| Metric | Winning-run pattern | Warning sign |
|---|---|---|
| `eval/avg_score` | Final arbiter; best run finished at 2140.1 | Training looks better, but eval stays flat or regresses |
| `train/avg_score` | Roughly 39.4 by update 10 | Stuck in low 30s after several updates |
| `train/entropy` | Stayed around 0.64 at update 10 | Early entropy collapse or manually forcing it lower hurt final eval |
| `train/clip_fraction` | Around 0.016 on the winning run | Sustained values above ~0.04 often preceded worse final results |
| `train/approx_kl` | Roughly 0.003 to 0.004 | Large spikes imply overly aggressive updates |
| `train/value_loss` | Fell to about 0.45 by update 10 | Large noisy swings usually came with weaker eval |

What this means in practice:
- Gate PPO changes on `eval/avg_score`, not just `train/avg_score`
- Treat `clip_fraction` and `approx_kl` as stability checks, not the target
- The best PPO run still looked active at the 30-minute cutoff, so longer
  training is probably somewhat helpful, but only for a stable config

---

## 4. Metrics to Track

### During Training
| Metric | What It Tells You | Healthy Sign |
|---|---|---|
| **Episode score** | Raw game performance | Trending upward |
| **Avg score (last 100)** | Smoothed learning progress | Monotonically increasing |
| **Epsilon** (DQN) | Exploration rate | Decaying from 1.0 → 0.001 |
| **Loss** | How wrong the network's predictions are | Decreasing, then stabilizing |
| **Episode length (steps)** | How long the agent survives | Should correlate with score |
| **Replay buffer size** | How much experience is stored | Should fill to max (200k) |

### During Evaluation
| Metric | What It Tells You | Target |
|---|---|---|
| **Avg score (30 games)** | Reliable performance estimate | > 1000 is good, > 1500 is strong |
| **Median score** | Less sensitive to outliers than mean | Close to or above mean |
| **Min score** | Worst-case performance (robustness) | > 200 means no catastrophic failures |
| **Max score** | Ceiling of capability | > 2000 means agent has learned timing well |
| **Std deviation** | Consistency | Lower is better; < 400 is reasonable |

### Comparing Algorithms
| Metric | Why It Matters |
|---|---|
| **Sample efficiency** | How many episodes to reach a given score (fewer = better) |
| **Stability** | Does performance oscillate or plateau reliably? |
| **Final performance** | Best average eval score achieved |
| **Training time** | Wall-clock seconds to convergence |
| **Hyperparameter sensitivity** | Does it work with default params or need extensive tuning? |

---

## 5. Current Results Benchmark

Results from training on the 1:1 Chromium-faithful Python environment:

### Educational Algorithms (2000 episodes each, simple envs)
| Algorithm | Best Score | Avg (last 100) | Notes |
|---|---|---|---|
| Value Network TD(0) | 497 | 63 | Prediction only, no control |
| REINFORCE | 414 | 103 | High variance, slow learning |
| REINFORCE + Baseline | 451 | 95 | Baseline helps somewhat |
| Actor-Critic (1-step) | 2771 | 52 | Occasional spike, unstable |
| Actor-Critic (N-step) | 369 | 87 | N-step didn't help here |
| TD(λ) | 414 | 86 | Eligibility traces |
| A2C | 398 | 76 | Synchronous variant |
| PPO | 439 | 97 | Clipping didn't add much here |

### DQN Variants (500 episodes each, educational)
| Variant | Best Score | Avg (last 100) |
|---|---|---|
| Vanilla DQN | 1710 | 188 |
| Double DQN | 656 | 150 |
| Dueling Double DQN | 573 | 177 |
| Dueling DDQN + PER | 763 | 180 |

### Production DQN (10000 episodes, tuned hyperparameters)
| | Score |
|---|---|
| **Eval avg (30 games)** | **1562** |
| Eval median | 1564 |
| Eval max | 2169 |
| Training time | ~45 min |

### Heuristic Agent (zero ML)
| | Score |
|---|---|
| **Avg (30 games)** | **780** |
| Median | 640 |
| Max | 1736 |
| Min | 183 |
| Training time | **0 seconds** |

**Key takeaway**: The heuristic agent gets 50% of the DQN's performance with zero training.
The DQN wins because it learns subtle timing that the heuristic's safety margins miss.

---

## 6. Research Questions to Explore

### "Do we even need ML for this?"
The Chrome Dino game has a **tiny effective state space**:
- ~300 useful distance values (0-300px at integer speed)
- 3 obstacle height classes (small cactus, large cactus, pterodactyl)
- ~70 relevant speed values (6.0 to 13.0 at 0.001 increments, but speed changes slowly)
- Binary: on ground or airborne (no decision while airborne)

That's **~63,000 decision-relevant states** — easily solvable with a lookup table.
In practice, a **210-entry table** (300 distances × 3 heights, bucketed by speed) would suffice.

**Exercise**: Modify `heuristic_agent.py` to build and use an explicit lookup table.
Compare its performance to the forward-simulation version.

### Why does the heuristic agent still lose?
The forward-simulation heuristic (avg 780) loses to DQN (avg 1562) because:
1. **Safety margins**: The heuristic uses conservative collision margins (3px extra) to avoid narrow-phase surprises. DQN learns the exact safe boundary.
2. **Multi-obstacle planning**: The heuristic only considers 2 obstacles ahead. At high speeds, 3+ obstacles matter.
3. **Speed-dependent timing**: The heuristic uses a fixed "250px look-ahead". DQN implicitly adjusts.

**Exercise**: Try tuning the heuristic's parameters (margin size, look-ahead distance, desperation threshold) and measure the impact.

### Reward shaping matters more than architecture
The biggest DQN improvement came from changing `reward = 0.1` to `reward = 0.01` per survival step (keeping crash penalty at -10). This is because:
- With +0.1/step: 100 steps of survival = +10, which cancels the -10 crash penalty
- With +0.01/step: 1000 steps of survival = +10, so crash penalty dominates for bad episodes
- The network can now clearly distinguish "survived 50 steps" (bad) from "survived 5000 steps" (good)

**Exercise**: Try different reward values (0.001, 0.05, 0.1, 1.0) and plot learning curves.

### On-policy vs off-policy
- **On-policy** (REINFORCE, Actor-Critic, PPO, A2C): Learn from current policy's experience. Must throw away old data.
- **Off-policy** (DQN): Learn from any past experience via replay buffer. Much more sample-efficient.

For the dino game, off-policy wins because the environment is deterministic — old experience doesn't go stale.

**Exercise**: Compare sample efficiency by plotting "score vs total environment steps" (not episodes) for DQN vs PPO.

### Why did Actor-Critic spike to 2771 but average 52?
One-step Actor-Critic occasionally stumbles into a good policy, then loses it because:
1. One-step TD has high bias (bootstraps from a bad value estimate)
2. No replay buffer — each transition is used once then discarded
3. Policy gradient has high variance with single-step returns

**Exercise**: Compare `actor_critic.py` (1-step) vs `actor_critic_nstep.py` (n-step) training curves in TensorBoard.

---

## 7. Suggested Study Schedule

### Week 1: Environment & Foundations
- [ ] Read Sutton & Barto Ch 1-3
- [ ] Read through `dino_rl/env.py` — understand the state space, actions, rewards
- [ ] Run `python -m dino_rl.heuristic_agent` — see what zero-ML achieves
- [ ] Read `dino_rl/heuristic_agent.py` — understand the forward simulation
- [ ] Compare env.py with original `offline.js` — understand the faithful port

### Week 2: Value Methods & TD Learning
- [ ] Read Sutton & Barto Ch 6 + 9.3
- [ ] Run `python -m dino_rl.algorithms.train_all --algo value_network`
- [ ] Read `value_network.py` — how does a network approximate V(s)?
- [ ] Run `python -m dino_rl.algorithms.train_all --algo td_lambda`
- [ ] Read `td_lambda.py` — how do eligibility traces help?
- [ ] Compare TD(0) vs TD(λ) in TensorBoard

### Week 3: Policy Gradients
- [ ] Read Sutton & Barto Ch 13
- [ ] Run and read: `reinforce.py`, `reinforce_baseline.py`
- [ ] Run and read: `actor_critic.py`, `actor_critic_nstep.py`
- [ ] Compare all four in TensorBoard
- [ ] Key question: why does adding a baseline help REINFORCE?

### Week 4: Deep Q-Networks
- [ ] Read the original DQN paper (Mnih 2013)
- [ ] Read about Double DQN, Dueling DQN, PER (papers above)
- [ ] Run `python -m dino_rl.algorithms.train_all --algo dqn`
- [ ] Read `dqn_progression.py` — all 4 variants in one file
- [ ] Run production training: `python -m dino_rl.train_dqn`
- [ ] Key question: why does DQN massively outperform policy gradient methods here?

### Week 5: Modern Methods & Analysis
- [ ] Read PPO and A2C papers
- [ ] Run and read: `ppo.py`, `a2c.py`
- [ ] Experiment with reward shaping (modify `train_dqn.py` reward values)
- [ ] Try the browser demo: `python -m dino_rl.play_browser`
- [ ] Write up: which algorithm would you choose for this problem, and why?

### Week 6: Go Deeper
- [ ] Tune the heuristic agent to beat DQN (is it possible?)
- [ ] Implement a new algorithm (e.g., SAC, Rainbow DQN, or TRPO)
- [ ] Try pixel-based observation instead of features
- [ ] Profile training: where is time spent? (environment steps vs gradient updates)

---

## 8. File Map

```
Reinforcement_Learning/
├── pyproject.toml              # Package definition, install with: pip install -e .
├── STUDY_PLAN.md               # This file
├── README.md                   # Project overview
│
├── dino_rl/                    # Main package
│   ├── env.py                  # 1:1 Chromium T-Rex Runner simulation
│   ├── common.py               # Shared training utilities (DinoFeatureEnv, plotting)
│   ├── networks.py             # DuelingDQN architecture (shared by train + play)
│   ├── train_dqn.py            # Production DQN training loop
│   ├── play_browser.py         # Control real Chrome browser with trained model
│   ├── heuristic_agent.py      # Zero-ML agent (forward simulation)
│   └── algorithms/             # 10 educational RL algorithms
│       ├── train_all.py        # Runner for all algorithms
│       ├── value_network.py    # Semi-gradient TD(0)
│       ├── reinforce.py        # REINFORCE (Monte Carlo policy gradient)
│       ├── reinforce_baseline.py
│       ├── actor_critic.py     # One-step TD actor-critic
│       ├── actor_critic_nstep.py
│       ├── td_lambda.py        # TD(λ) with eligibility traces
│       ├── dqn_progression.py  # Vanilla → Double → Dueling → PER DQN
│       ├── ppo.py              # Proximal Policy Optimization
│       └── a2c.py              # Advantage Actor-Critic
│
├── checkpoints/
│   └── dino_runner.pth         # Trained DQN model weights
│
└── results/                    # Training outputs
    ├── *.json                  # Score histories per algorithm
    ├── *.png                   # Training curve plots
    └── runs/                   # TensorBoard logs
```
