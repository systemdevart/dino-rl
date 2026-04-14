# autoresearch

This is an experiment to have the LLM do its own research on training a PPO agent to play the Chrome Dino game.

## Context

The repo contains a **1:1 faithful Python simulation** of the Chromium T-Rex Runner (`dino_rl/env.py`), plus 10 educational RL algorithms and a production DQN. The current best agents:

| Agent | Avg Score | Notes |
|---|---|---|
| DQN (Dueling Double) | 1,562 | 30-game eval, ~45 min training |
| Heuristic (zero ML) | 780 | Forward simulation, 0 training |
| PPO (current) | ~30 | Barely above random — needs work |

**The goal: train PPO to achieve eval avg >= 10,000.** This requires ~32,600 frames of perfect play at max speed 13.0.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these for full context:
   - `dino_rl/algorithms/ppo.py` — the file you modify. Network architecture, hyperparameters, PPO update, rollout collection, training loop.
   - `dino_rl/common.py` — shared env wrapper (`DinoFeatureEnv`), evaluation, plotting, TensorBoard writer. Read-only reference.
   - `dino_rl/env.py` — the game simulation. Read-only reference. Understand the physics, scoring (`round(distance * 0.025)`), and feature extraction (8-dim).
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training runs for a **fixed time budget of 30 minutes** (wall clock). You launch it as:

```
CUDA_VISIBLE_DEVICES=1 python -m dino_rl.algorithms.train_all --algo ppo
```

**What you CAN modify:**
- `dino_rl/algorithms/ppo.py` — this is the primary file you edit. Everything is fair game: network architecture (width, depth, activation), hyperparameters (LR, clip_eps, GAE lambda, entropy coeff, rollout length, minibatch size, PPO epochs), training loop structure, reward shaping override, learning rate schedules, gradient clipping, observation normalization, etc.

**What you CANNOT modify:**
- `dino_rl/env.py` — the game simulation is fixed. It is the ground truth.
- `dino_rl/common.py` — the evaluation harness and env wrapper are fixed.
- Do not install new packages. Use only what's in `pyproject.toml`.

**The goal is simple: get eval avg score >= 10,000.** The time budget is fixed at 30 minutes. Everything in `ppo.py` is fair game.

**Key technical facts:**
- Score 10k requires ~32,600 frames of survival. Rollout length must be >= 33,000 to experience the full game.
- Speed ramps from 6.0 to 13.0 (max) over ~7,000 frames. At max speed, obstacles arrive every ~8-10 frames with tight gaps.
- The agent needs 99.97%+ per-obstacle accuracy to reliably score 10k (one mistake = game over).
- The 8-dim feature vector is: [obstacle1_dist, obstacle1_width, obstacle1_height, obstacle2_dist, dino_height, jump_velocity, speed, on_ground].
- Action space: 2 actions (0=do nothing, 1=jump). Max entropy = ln(2) ≈ 0.693.
- Reward: +0.01/step survival, -10.0 on crash (set in `common.py`).

**The first run**: Your very first run should always be to establish the baseline with current settings, so you will run the training script as is.

## Output format

The training script prints periodic updates:

```
Update  10/5000 | Steps  400000 | Episodes  45 | AvgScore   123.4 | PolicyL -0.0123 | ValueL  2.345 | Entropy 0.5432 | KL 0.00345 | Clip 0.123
  >> Eval @ update 50: avg=456.7  min=123  max=890
```

View training curves in TensorBoard:

```
tensorboard --logdir results/runs
```

Key metrics to watch:
- `eval/avg_score` — the target metric. Must reach 10,000.
- `train/avg_score` — training performance (noisy but should trend up).
- `train/entropy` — policy randomness. Should decrease from ~0.69 as agent learns, but not collapse to 0.
- `train/clip_fraction` — how often PPO clipping activates. Healthy: 0.05-0.2.
- `train/approx_kl` — policy divergence per update. Healthy: < 0.05.
- `train/value_loss` — critic accuracy. Should decrease over time.
- `train/advantage_std` — signal quality. Should be > 0, not collapsing.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	eval_avg	eval_max	status	description
```

1. git commit hash (short, 7 chars)
2. eval_avg score achieved (e.g. 1234.5) — use 0.0 for crashes
3. eval_max score achieved (e.g. 5678) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	eval_avg	eval_max	status	description
a1b2c3d	32.9	65	keep	baseline (rollout=4096 128x128 net)
b2c3d4e	456.7	1234	keep	rollout=40000 256x256x128 net
c3d4e5f	0.0	0	crash	rollout=80000 OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr14`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `ppo.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `CUDA_VISIBLE_DEVICES=1 python -m dino_rl.algorithms.train_all --algo ppo > run.log 2>&1`
5. Read out the results: `tail -5 run.log` and `grep "Eval\|TARGET\|TIME BUDGET" run.log | tail -10`
6. If the grep output is empty or shows errors, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If eval_avg improved (higher), you "advance" the branch, keeping the git commit
9. If eval_avg is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~30 minutes. If a run exceeds 40 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix it and re-run. If the idea itself is fundamentally broken, log "crash", and move on.

**Ideas to try** (in rough priority order):
- Hyperparameter sweeps: LR, clip_eps, entropy_coeff, GAE lambda
- Network architecture: deeper/wider, separate actor-critic networks, layer norm
- Rollout length tuning: balance between seeing full game and update frequency
- PPO epochs and minibatch size: more epochs = more sample reuse but risk overfitting
- Observation normalization (running mean/std of features)
- Reward shaping: different survival reward, speed-based bonus
- Learning rate schedules: cosine, warmup + decay
- Gradient norm clipping threshold
- Orthogonal weight initialization
- Value function clipping

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
