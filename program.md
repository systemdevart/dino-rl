# Dino RL Browser 5k Research Manifest
**Status:** complete and frozen. Last updated 2026-07-26.

This repository produced three DQN policies that consistently exceed score
5,000 in the real Chrome Dino game. The promoted checkpoints are post-physics-
correction artifacts and pass the same independent cold-browser protocol.

## Completion Gate

Formal verification is unchanged:

- Real `chrome://dino` through Selenium/ChromeDriver.
- Deterministic pause-and-frame-step environment, one frame per action.
- Normal game starts; training speed curricula are disabled.
- Greedy policy, 8 independent episodes, 22,000-step cap.
- Pass only when the minimum of all 8 scores is at least 5,000.
- In-loop evaluations are candidate selection only, never formal proof.

| Track | Observation and provenance | Promoted checkpoint | Formal result |
|---|---|---|---|
| Feature DQN | 10 engineered features; browser RL lineage; no demonstrations | `checkpoints/dino_browser_dqn_speeddropm_candidate.pth` | 8/8; min 6375; mean 6484 |
| Image DQfD | 4x84x336 pixels; protected corrected demo bank | `checkpoints/dino_dqn_image_dqfdfixc_candidate.pth` | 8/8; all 6500 |
| Image pure RL | 4x84x336 pixels; no teacher/demo/external prefill | `checkpoints/dino_dqn_image_scratchfixw_candidate.pth` | 8/8; all 6500 |

Exact formal scores:

- Feature: `[6375, 6500, 6500, 6500, 6500, 6500, 6500, 6500]`
- Both image tracks: `[6500, 6500, 6500, 6500, 6500, 6500, 6500, 6500]`

Artifact integrity:

| Artifact | SHA-256 |
|---|---|
| `dino_browser_dqn_speeddropm_candidate.pth` | `6691cd1b7fd7ebcdcc66be86d3dbdb0f5898367d5c9cb1383810ed5e348491e0` |
| `dino_dqn_image_dqfdfixc_candidate.pth` | `80cb6dbf0dc0fd0bca075d230f5817ac7ebecf96085dd0a1e832ac39019e3913` |
| `dino_dqn_image_scratchfixw_candidate.pth` | `e317c8063a407cbcc7e5ef19404ecaa88a4533b09a5b63af1926042dbecab728` |

The compact formal outputs are tracked under `results/formal/`. Full local logs
remain beside the retained artifacts.

## Validity Boundary

On 2026-07-13, the speed-drop contract was corrected. Chrome's
`setSpeedDrop()` sets both `speedDrop=true` and `jumpVelocity=1`; the old
bridge and Python simulator set only the flag. `dino_rl/browser_env.py` and
`dino_rl/env.py` now match Chrome.

Pre-correction checkpoints, replay buffers, demo banks, and scores are not
valid evidence for this gate. In particular, do not substitute
`browser_dqn_buffer.pkl`, `imgbuf_scratch.pt`, `imgbuf_dqfd.pt`, or
`image_dqn_bank.npz` for the corrected `*speeddrop*`, `*fix*`, and
`image_dqn_bank_clean.npz` artifacts.

## Final Methods

The browser environment uses a virtual clock and explicit `Runner.update()`
steps, so screenshot and inference latency cannot create action drift. All
policies decide every frame (`fpa=1`).

**Feature DQN:**
- Dueling DQN, 663,300 parameters, 12 parallel browser environments.
- Browser-generated replay only; no demonstrations.
- Obstacle-clear reward, n-step returns, speed curriculum, no-duck actions,
  persistent 200k replay, slow target update, and lower-tail selection.
- M continued J with a gentle sequence-jump regularizer and passed formally.

**Image DQfD:**
- IMPALA residual encoder plus dueling head, 7.67M parameters, Double DQN.
- Pixel policy input is a 4x84x336 grayscale stack.
- 32 parallel browsers, AMP, GPU-resident 120k live replay.
- Separate protected 120k corrected heuristic-demo replay, 50% batch fraction,
  and DQfD large-margin loss. The demo bank is never evicted.

**Image pure RL:**
- Same pixel-only IMPALA/dueling family, 32 browsers, AMP, 360k live replay.
- No teacher data, demo bank, margin imitation, or external prefill.
- A protected 30k anchor was extracted from the policy's own corrected replay
  and used for TD updates only. Its provenance records
  `teacher_or_demo_data=False`.
- Paired pre-action features shape clear/far-jump reward only; they are never
  policy observations. W resumed T's full live replay and Adam state.

The main reusable findings are persistent live replay, lower-tail checkpoint
selection by `(eval min, eval avg)`, explicit LR override on resume, protected
DQfD data, and speed-start curriculum to expose max-speed image states.

## Measured Reproduction Time

The timing terms are distinct:

- **Winning continuation:** final command from its retained precursor state.
- **Corrected lineage:** actual successful post-correction ancestry. It starts
  from same-track pre-correction weights, though feature/pure provenance is RL.
- **Strict cold start:** random weights and empty live replay; never measured
  end-to-end with the final recipes, so its numbers are estimates only.

### Winning Continuations

| Track | Candidate wall time | Formal wall time | Practical launch-to-proof |
|---|---:|---:|---:|
| Feature M from J | 23m42s | 7m14s measured | 33m33s observed |
| DQfD fixc from fix-best/replay | 43m35s | about 19-26m | about 1.05-1.16h if stopped at candidate; 1.52h observed |
| Pure W from T live state | 22m50s including one immediate OOM/relaunch | at most 25m51s | 48m41s observed |

Only the feature track retains both a precursor model and compatible replay for
a warm follow-up. The original optimizer, RNG state, and start-of-run replay
snapshot were not retained, so no winning run has an exact continuation.
The image times describe historical runs whose mutable live replay was deleted.
Image verifier start timestamps were not preserved in their logs. The 19-26
minute range is bounded from candidate/trainer/result timestamps. Feature
verification is directly timed. Learner markers such as `[480s]` and `[600s]`
exclude the following 8-episode in-loop evaluation and are not total wall time.

### Actual Corrected Lineages

| Track | Successful active wall | New corrected env steps | Calendar span to proof |
|---|---:|---:|---:|
| Feature | about 7.45h | about 5.18M | 33.24h |
| DQfD | about 1.8-2.4h, depending on whether the fix2 polish is replayed | about 0.78M | 3.27h from clean-bank launch |
| Pure image | about 24.8h | about 10.66M | 32.35h |

The larger calendar spans include gaps, concurrent branches, artifact
snapshots, diagnosis, and failed controls. They should not be added to the
last-mile estimates.

### Strict Cold-Start Planning Estimates

These are capacity-planning ranges, not measured reproduction claims:

| Track | Estimated one-seed wall time | Basis |
|---|---:|---|
| Feature | 24-48h | Historical random browser lineage climbed on day 1 and capped on day 2 |
| DQfD | 24-48h after/including a clean bank | Final curriculum capped within a day, but corrected winner reused a trained encoder |
| Pure image | 48-72h | Even the warm corrected lineage used 24.8 active and 32.35 calendar hours |

Stochastic RL may require another seed. A multi-seed research reproduction
multiplies accelerator-hours even when seeds run concurrently. There is no
validated strict-random one-command recipe, so this manifest does not invent
one.

## Hardware and Storage
| Track | Proven training setup | Peak mutable data | Current retained assets |
|---|---|---|---|
| Feature | 12 Chrome envs; one CUDA device; CPU eval clone | 32.7MB per replay | 86.6MB |
| DQfD | 32 Chrome envs; one B200 | 27.22GB live snapshot plus 1.10GB compressed bank | about 1.16GB; live snapshot deleted |
| Pure image | 32 Chrome envs; one 178GB B200 | 81.42GB live snapshot plus 6.77GB self bank | about 6.84GB; live snapshot deleted |

The pure-image OOM log shows one 37.85GiB replay tensor allocation failing
with only 35.33GiB free; the paired live tensors need about 75.7GiB before
model, optimizer, batches, and the protected self bank. DQfD similarly holds
both live and protected demo replay on GPU. These minimums are inferred, not
benchmarked on smaller accelerators.

Formal verification needs 8 fresh browsers. Feature verification can be
CPU-only; image verification uses a spare GPU but does not allocate replay.
Do not run multiple 32-browser jobs unless the host has sufficient CPU, RAM,
file descriptors, and Chrome process capacity.

## Retained Artifacts
The storage audit retains exactly 10 research assets:

- Feature (4): `dino_browser_dqn_speeddrop{j,m}_candidate.pth` and
  `browser_dqn_buffer_speeddrop_{j,m}.pkl`.
- DQfD (3): `dino_dqn_image_dqfdfix_best.pth`,
  `dino_dqn_image_dqfdfixc_candidate.pth`, and `image_dqn_bank_clean.npz`.
- Pure (3): `dino_dqn_image_scratchfix{t,w}_candidate.pth` and
  `imgbuf_scratchfixi_self30k.pt`.

Also retain the three formal logs and `checkpoints/README.md`.

Candidate-only evaluation does not need live replay snapshots or demo/self
banks. `imgbuf_dqfdfixc.pt` and `imgbuf_scratchfixt.pt` were mutable 27.22GB
and 81.42GB snapshots and have been deleted. Other intermediates are not
completion evidence.

## Reproduction Commands
Run commands from the repository root and select idle CUDA devices. The feature
command is a runnable warm follow-up using retained state, not an exact replay
of the measured winning run.

### Feature Warm Follow-up
```bash
DQN_TAU=0.001 DQN_LR=0.000005 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> \
python -u train_browser_dqn.py \
  --num-envs 12 --updates-per-tick 1 --n-step 16 --terminal-frac 0.10 \
  --sequence-replay-frac 0.005 --speed-curriculum 0.5 --no-duck \
  --empty-state-noop-margin 0.02 --empty-state-noop-weight 0.25 \
  --far-state-noop-distance 200 --far-state-noop-margin 0.02 --far-state-noop-weight 0.25 \
  --sequence-jump-margin 0.02 --sequence-jump-weight 0.005 \
  --time-budget-sec 86400 --eval-every-sec 480 --eval-episodes 8 --target 5000 \
  --init-from checkpoints/dino_browser_dqn_speeddropj_candidate.pth \
  --eps-start 0.01 --eps-min 0.01 --eps-decay 1.0 --min-replay 5000 \
  --buffer-file checkpoints/browser_dqn_buffer_speeddrop_m.pkl \
  --anneal-after 99999 --out checkpoints/dino_browser_dqn_speeddropm.pth
```

This combines the J precursor weights with the compatible final M replay and
starts fresh optimizer and RNG state. The measured M continuation took 23m42s,
but this retained-state recipe has not been timed or formally revalidated.

### DQfD Historical Winner and Warm Restart
This is the exact winning argv, retained as provenance. Its named live replay
was deleted, so running it now would silently start an empty replay and would
not reproduce fixc. A warm restart may use the retained fix-best checkpoint
and clean bank with a new buffer path, but its time and result are unvalidated.

```bash
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> python -u image_dqn.py \
  --num-envs 32 --encoder impala --image-size 84 --image-width 336 \
  --deterministic --frames-per-action 1 --n-step 3 --amp --buffer 120000 \
  --bank checkpoints/image_dqn_bank_clean.npz --margin 0.8 \
  --margin-weight 1.0 --demo-frac 0.5 \
  --init-from checkpoints/dino_dqn_image_dqfdfix_best.pth \
  --milestone-bonus 1.0 --buffer-file checkpoints/imgbuf_dqfdfixc.pt \
  --speed-curriculum 0.5 --lr 1e-5 \
  --eps-start 0.02 --eps-end 0.02 --eps-decay-steps 1000 \
  --terminal-frac 0.02 --anneal-after 5000 --time-budget-sec 86400 \
  --eval-every-sec 600 --eval-episodes 8 --run-tag dqfdfixc
```

For a new bank, use the promoted corrected feature policy as teacher. This
does not recreate the retained bank byte-for-byte and requires new validation:

```bash
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> python -u harvest_image_bank.py \
  --teacher checkpoints/dino_browser_dqn_speeddropm_candidate.pth \
  --num-envs 12 --target 120000 --epsilon 0 \
  --speed-curriculum 0.5 --milestone-bonus 1.0 \
  --out <new-bank.npz> --max-seconds 1800
```

### Pure-Image Historical Winner
This exact W argv required T's deleted live replay and optimizer snapshot.
The T candidate and self bank remain, but a prior checkpoint-only restart with
empty live replay collapsed to score 42 on its first evaluation. No verified
pure-image warm restart is currently retained; use the 48-72h cold estimate.

```bash
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> python -u image_dqn.py \
  --num-envs 32 --encoder impala --image-size 84 --image-width 336 \
  --deterministic --frames-per-action 1 --n-step 10 --amp --buffer 360000 \
  --self-bank checkpoints/imgbuf_scratchfixi_self30k.pt --self-frac 0.125 \
  --clear-bonus 1 --far-jump-penalty 0.10 --far-jump-distance 200 \
  --buffer-file checkpoints/imgbuf_scratchfixt.pt --resume-live \
  --speed-curriculum 0.05 --lr 5e-7 --tau 0.0002 \
  --eps-start 0 --eps-end 0 --eps-decay-steps 1000 --terminal-frac 0.01 \
  --anneal-after 0 --time-budget-sec 86400 \
  --eval-every-sec 600 --eval-episodes 8 --run-tag scratchfixw
```

### Formal Verification and Integrity

```bash
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES="" python -u eval_browser_dqn.py \
  checkpoints/dino_browser_dqn_speeddropm_candidate.pth 8 22000
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> python -u verify_image.py \
  checkpoints/dino_dqn_image_dqfdfixc_candidate.pth 8 22000
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=<gpu> python -u verify_image.py \
  checkpoints/dino_dqn_image_scratchfixw_candidate.pth 8 22000
sha256sum checkpoints/dino_browser_dqn_speeddropm_candidate.pth \
  checkpoints/dino_dqn_image_dqfdfixc_candidate.pth \
  checkpoints/dino_dqn_image_scratchfixw_candidate.pth
```

Keep the promoted candidates immutable. Before any browser cleanup, confirm no
trainer or verifier owns Chrome processes. Browser recovery is per environment;
an infrastructure retry is not a policy failure.
