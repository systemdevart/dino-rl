# autoresearch

This experiment asks the LLM to improve the **browser-image PPO** agent for
Chrome Dino.

## Context

The repo now has two distinct PPO tracks:

| Track | Backend | Observation | Status |
|---|---|---|---|
| Feature PPO | Python simulator | 10 engineered features | Strong; already far ahead |
| Image PPO | Real Chrome + ChromeDriver | 4 x 84 x 84 grayscale stack | Current research target |

This file is about the **image PPO** track only.

Current browser-image setup:

- Environment backend: real `chrome://dino` through `dino_rl/browser_env.py`
- Observation: 4 stacked grayscale frames of shape `(4, 84, 84)`
- Preprocessing: crop to gameplay strip, mask score HUD, max-pool recent frames
- Action repeat: `4`
- Action space: `3` actions
  - `0 = noop`
  - `1 = jump`
  - `2 = duck / hold down`
- Reward:
  - positive reward from scaled `distanceRan` progress
  - `-1.0` on crash
- Browser recovery:
  - the env is expected to recover from dead sessions / tab crashes instead of
    killing the trainer

Current checkpoint baseline:

- Best saved browser-image checkpoint:
  `checkpoints/dino_ppo_browser_image_best.pth`
- Best known eval from that line: `avg=75.7`, `min=57`, `max=99`
- A longer continuation run was manually stopped after plateauing and did not
  beat that checkpoint

## Goal

The current target for browser-image PPO is:

- primary target: `eval avg >= 5,000`
- intermediate milestone: `eval avg >= 1,000`

This is a hard sample-efficiency problem. Do not assume image PPO will behave
like the simulator feature agent.

## Setup

To start a new research pass, work with the user to:

1. Propose a run tag based on the date, for example `apr20-image`
2. Create a fresh branch `autoresearch/<tag>` from current `master`
3. Read the in-scope files:
   - `dino_rl/algorithms/ppo.py`
   - `dino_rl/browser_env.py`
   - `dino_rl/play_browser.py`
   - `dino_rl/policy_loader.py`
4. Confirm that Chrome/ChromeDriver browser control is functional
5. Kick off the experiment loop

## Experimentation

Each experiment uses a single GPU and runs for a **fixed wall-clock budget of 1
hour**.

Launch command from repo root:

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m dino_rl.algorithms.ppo \
  --env-backend browser \
  --observation-mode image \
  --time-budget-sec 3600 \
  --eval-every 5 \
  --print-every 1 \
  > run_browser_image.log 2>&1
```

If a best browser-image checkpoint already exists, resume from it unless there
is a good reason not to:

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m dino_rl.algorithms.ppo \
  --env-backend browser \
  --observation-mode image \
  --time-budget-sec 3600 \
  --eval-every 5 \
  --print-every 1 \
  --init-checkpoint checkpoints/dino_ppo_browser_image_best.pth \
  > run_browser_image.log 2>&1
```

### What you CAN modify

- `dino_rl/algorithms/ppo.py`
- `dino_rl/browser_env.py`
- `dino_rl/play_browser.py`
- `dino_rl/policy_loader.py`

Everything needed for image PPO is fair game: CNN architecture, PPO
hyperparameters, rollout length, minibatch size, entropy coefficient, clipping,
reward scaling, evaluation cadence, browser observation preprocessing, action
repeat, recovery behavior, and checkpointing.

### What you CANNOT modify

- `dino_rl/env.py` for the purpose of making the browser-image task easier
- install new packages
- silently switch the experiment back to simulator features

The task here is specifically to improve PPO on **browser images**.

## Key Technical Facts

- Action space is `3`, so max entropy is `ln(3) ~= 1.099`
- Browser-image observations are expensive; update cadence is much slower than
  simulator PPO
- Current image defaults are tuned separately from feature PPO:
  - `rollout_len = 512`
  - `minibatch_size = 64`
  - `eval_every = 5`
  - `score_delta_coeff = 0.0`
- The old future auxiliary loss is gone; this image path is now plain PPO with
  a CNN encoder
- Browser failures such as `tab crashed`, disconnected sessions, or dead
  ChromeDriver connections should be treated as recoverable env failures, not as
  acceptable reasons for the trainer to exit

## Output Format

The trainer prints periodic updates like:

```text
Update   40/100000 | Steps   20480 | Episodes   6 | AvgScore    54.3 | PolicyL  0.0012 | ValueL 121.337 | Entropy 0.9812 | KL 0.00234 | Clip 0.154
  >> Eval @ update 40: avg=72.7  min=58  max=96
```

View TensorBoard with:

```bash
tensorboard --logdir results/runs
```

Key metrics to watch:

- `eval/avg_score` — target metric
- `train/avg_score` — noisy, but should trend up over time
- `train/entropy` — if it stays near `1.099`, the policy is close to random
- `train/clip_fraction` — healthy range is usually around `0.05` to `0.25`
- `train/approx_kl` — should stay controlled, usually `< 0.05`
- `train/value_loss` — critic fit
- browser recovery messages in the log — useful to track instability in Chrome

## Logging Results

When an experiment finishes, log it to `results.tsv` as tab-separated text:

```text
commit	eval_avg	eval_max	status	description
```

Columns:

1. short git commit hash
2. `eval_avg`
3. `eval_max`
4. status: `keep`, `discard`, or `crash`
5. short description of the image/browser experiment

Do not commit `results.tsv`.

## The Experiment Loop

Run on a dedicated branch such as `autoresearch/apr20-image`.

Loop:

1. Inspect the current branch and commit
2. Make one focused image-PPO change
3. Commit the change
4. Run a 1-hour browser-image PPO experiment
5. Read the results from the log
6. If the run crashed:
   - inspect the traceback
   - distinguish between a browser-recovery bug, a trainer bug, and a bad idea
7. Record the result in `results.tsv`
8. If `eval_avg` improved, keep the commit and keep the new best checkpoint
9. If `eval_avg` did not improve, revert the code change and continue from the
   last good point

### Timeout rule

- Budget per experiment: `3600` seconds
- If a run goes materially past `1 hour` without the trainer honoring the time
  budget, stop it manually and treat that as a failure in the experiment loop

### Crash rule

- A recoverable Chrome crash should not kill PPO
- If PPO still exits on browser failure, fix that first before trusting any
  learning result

### Never confuse tracks

The feature agent and the image agent are different experiments with different
budgets, observations, and expectations. Improvements on feature PPO do not
count as progress for this browser-image program.
