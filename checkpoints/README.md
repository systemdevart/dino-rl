# Local Research Artifacts

Model and replay binaries are intentionally excluded from Git. The following
local candidates passed the formal real-browser protocol: eight independent
greedy episodes, normal starts, one frame per action, and a 22,000-step cap.

| Track | Local artifact | SHA-256 | Formal result |
|---|---|---|---|
| Feature DQN | `dino_browser_dqn_speeddropm_candidate.pth` | `6691cd1b7fd7ebcdcc66be86d3dbdb0f5898367d5c9cb1383810ed5e348491e0` | 8/8, min 6375, mean 6484 |
| Image DQN, pure RL | `dino_dqn_image_scratchfixw_candidate.pth` | `e317c8063a407cbcc7e5ef19404ecaa88a4533b09a5b63af1926042dbecab728` | 8/8, all 6500 |
| Image DQN, DQfD | `dino_dqn_image_dqfdfixc_candidate.pth` | `80cb6dbf0dc0fd0bca075d230f5817ac7ebecf96085dd0a1e832ac39019e3913` | 8/8, all 6500 |

The minimum retained local training data is:

- Feature continuation:
  - `dino_browser_dqn_speeddropj_candidate.pth`
  - `browser_dqn_buffer_speeddrop_j.pkl`
  - `browser_dqn_buffer_speeddrop_m.pkl`
- DQfD restart:
  - `dino_dqn_image_dqfdfix_best.pth`
  - `image_dqn_bank_clean.npz`
    (`SHA-256 8f639540b7185ff46c0cbb32433964b575d28ed6d6184214daa4fb1ad2726954`)
- Pure-RL restart:
  - `dino_dqn_image_scratchfixt_candidate.pth`
  - `imgbuf_scratchfixi_self30k.pt`
    (`SHA-256 42ec00dde392552eb38c03f830400a7c25d325c3d73dcba9104ba935579e56a0`)

All `imgbuf_*.pt` live replay snapshots, intermediate checkpoints, and failed
experiment artifacts are disposable after a run. They are not required to
evaluate the promoted candidates. The retained restart assets do not make the
original stochastic trajectory byte-reproducible; mutable live replay and
optimizer snapshots were intentionally deleted to reclaim space.

Compact formal outputs are versioned under `results/formal/`. The checkpoint
binaries are published separately from Git so their hashes can be reviewed
without adding large objects to repository history.
