"""Image-based Dueling Double DQN for browser Dino.

Motivation: PPO is a refinement method -- strong at polishing an already-competent
policy (RLHF), weak at DISCOVERY under hard exploration (precise jump timing).
DQN is off-policy with a replay buffer, so a rare well-timed jump is stored and
replayed thousands of times until learned -- the user's feature-DQN reached >50k
this way. This ports that recipe to the generalizable IMAGE observation (84x336).

Defines self-contained Nature/IMPALA encoders plus a dueling head, uses the
feature DQN's reward scale (+0.01 alive / -10 crash), Double DQN targets, and
a soft-updated target network. Experience is collected from many parallel
browser workers into one shared replay buffer for throughput.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dino_rl.browser_env import VecChromeDinoImageEnv
from dino_rl.feature_contract import FEATURE_DIM, GAME_WIDTH


IMAGE_CHECKPOINT_FORMAT = "dino_image_dqn_v1"
IMAGE_ACTION_NAMES = ("noop", "jump", "duck")


class _ResidualBlock(nn.Module):
    """Pre-activation residual block used by the IMPALA encoder."""

    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv0(torch.relu(x))
        y = self.conv1(torch.relu(y))
        return x + y


class _ImpalaConvSequence(nn.Module):
    """Convolution, pooling, and two residual blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = _ResidualBlock(out_channels)
        self.res1 = _ResidualBlock(out_channels)

    def forward(self, x):
        x = self.pool(self.conv(x))
        return self.res1(self.res0(x))


def build_impala_cnn(in_channels, channels=(16, 32, 32)):
    """Build the IMPALA-style encoder used by the promoted image agents."""
    layers = []
    current_channels = in_channels
    for out_channels in channels:
        layers.append(_ImpalaConvSequence(current_channels, out_channels))
        current_channels = out_channels
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def build_nature_cnn(in_channels):
    """Build the original three-layer Nature DQN encoder."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
    )


def save_checkpoint(
    path,
    model,
    optimizer=None,
    *,
    update=None,
    eval_result=None,
    env_backend="browser",
    observation_mode="image",
    state_shape=None,
    encoder=None,
    action_names=None,
    training_config=None,
    provenance=None,
    **metadata,
):
    """Persist an image-DQN checkpoint without depending on PPO internals."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    resolved_shape = state_shape or getattr(model, "state_shape", None)
    resolved_encoder = encoder or getattr(model, "encoder_name", None)
    resolved_actions = tuple(action_names or IMAGE_ACTION_NAMES[
        :model.advantage.out_features
    ])
    if len(resolved_actions) != model.advantage.out_features:
        raise ValueError("action_names must match the model action count")
    payload = dict(metadata)
    payload.update({
        "checkpoint_format": IMAGE_CHECKPOINT_FORMAT,
        "algo": "dqn",
        "action_size": model.advantage.out_features,
        "action_names": resolved_actions,
        "env_backend": env_backend,
        "observation_mode": observation_mode,
        "model_state_dict": model.state_dict(),
    })
    if resolved_shape is not None:
        payload["state_shape"] = tuple(resolved_shape)
    if resolved_encoder is not None:
        payload["encoder"] = resolved_encoder
    if training_config is not None:
        payload["training_config"] = dict(training_config)
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if update is not None:
        payload["update"] = update
    if eval_result is not None:
        payload["eval_result"] = eval_result
    torch.save(payload, path)


class DuelingCNNQ(nn.Module):
    """Dueling DQN with a CNN encoder. Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, state_shape, action_size, encoder="impala"):
        super().__init__()
        if encoder not in {"impala", "nature"}:
            raise ValueError(f"unsupported image encoder: {encoder}")
        self.state_shape = tuple(state_shape)
        self.encoder_name = encoder
        c, h, w = state_shape
        self.encoder = (build_impala_cnn(c) if encoder == "impala"
                        else build_nature_cnn(c))
        with torch.no_grad():
            conv_dim = self.encoder(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_dim, 512), nn.ReLU())
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        f = self.fc(self.encoder(x).flatten(1))
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


@torch.no_grad()
def project_no_duck_advantage(model, margin):
    """Make raw action 2 strictly worse than noop for every input."""
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("no_duck_margin must be positive")
    advantage = model.advantage
    if advantage.out_features < 3:
        raise ValueError("no_duck requires a three-action DQN")
    advantage.weight[2].copy_(advantage.weight[0])
    advantage.bias[2].copy_(advantage.bias[0] - margin)


def greedy_action_indices(q_values, no_duck=False):
    """Return greedy action indices from the configured action set."""
    if no_duck:
        if q_values.shape[-1] < 3:
            raise ValueError("no_duck requires three-action Q-values")
        return q_values[..., :2].argmax(dim=-1)
    return q_values.argmax(dim=-1)


class ReplayBuffer:
    """Circular buffer; stores frames as uint8 to fit images in RAM."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add_batch(self, obs, actions, rewards, next_obs, dones, disc):
        n = len(actions)
        obs_u8 = (obs * 255).astype(np.uint8)
        next_u8 = (next_obs * 255).astype(np.uint8)
        for i in range(n):
            j = self.idx
            self.obs[j] = obs_u8[i]
            self.next_obs[j] = next_u8[i]
            self.actions[j] = actions[i]
            self.rewards[j] = rewards[i]
            self.dones[j] = dones[i]
            self.disc[j] = disc[i]
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device, terminal_frac=0.0,
               action=None, action_frac=0.0):
        n_terminal = min(int(batch_size * terminal_frac), batch_size)
        n_action = (min(int(batch_size * action_frac),
                        batch_size - n_terminal)
                    if action is not None else 0)
        selected = []
        if n_terminal:
            terminal_ids = np.flatnonzero(self.dones[:self.size] > 0)
            if len(terminal_ids):
                selected.append(np.random.choice(
                    terminal_ids, size=n_terminal, replace=True))
        if n_action:
            action_ids = np.flatnonzero(
                self.actions[:self.size] == int(action))
            if len(action_ids):
                selected.append(np.random.choice(
                    action_ids, size=n_action, replace=True))
        n_random = batch_size - sum(len(group) for group in selected)
        selected.insert(
            0, np.random.randint(0, self.size, size=n_random))
        ids = np.concatenate(selected)
        to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
        return (
            to(self.obs[ids].astype(np.float32) / 255.0, torch.float32),
            to(self.actions[ids], torch.long),
            to(self.rewards[ids], torch.float32),
            to(self.next_obs[ids].astype(np.float32) / 255.0, torch.float32),
            to(self.dones[ids], torch.float32),
            to(self.disc[ids], torch.float32),
        )


class GPUReplayBuffer:
    """Replay buffer kept entirely on the GPU as uint8.

    Profiling showed CPU-buffer random-gather (~649 ms / 512-sample) was ~80% of
    the DQN wall-clock. Holding the buffer on the GPU makes sampling ~0.5 ms
    (1270x faster) and removes the per-step host->device copy. Costs ~27 GB of
    VRAM for a 120k buffer at 4x84x336, which fits a B200 with room to spare.
    """

    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        # Per-transition bootstrap discount (gamma^n for n-step returns).
        self.disc = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.idx = 0
        self.size = 0

    def add_batch(self, obs, actions, rewards, next_obs, dones, disc):
        n = len(actions)
        # Write the N new transitions into a (possibly wrapping) index range.
        ids = (self.idx + np.arange(n)) % self.capacity
        ids_t = torch.as_tensor(ids, device=self.device)
        self.obs[ids_t] = torch.as_tensor(
            (obs * 255).astype(np.uint8), device=self.device)
        self.next_obs[ids_t] = torch.as_tensor(
            (next_obs * 255).astype(np.uint8), device=self.device)
        self.actions[ids_t] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.rewards[ids_t] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[ids_t] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.disc[ids_t] = torch.as_tensor(disc, dtype=torch.float32, device=self.device)
        self.idx = int((self.idx + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, device=None, terminal_frac=0.0,
               action=None, action_frac=0.0):
        n_terminal = min(int(batch_size * terminal_frac), batch_size)
        n_action = (min(int(batch_size * action_frac),
                        batch_size - n_terminal)
                    if action is not None else 0)
        selected = []
        if n_terminal:
            terminal_ids = torch.nonzero(
                self.dones[:self.size] > 0, as_tuple=False
            ).flatten()
            if terminal_ids.numel():
                selected.append(terminal_ids[torch.randint(
                    0, terminal_ids.numel(), (n_terminal,), device=self.device
                )])
        if n_action:
            action_ids = torch.nonzero(
                self.actions[:self.size] == int(action), as_tuple=False
            ).flatten()
            if action_ids.numel():
                selected.append(action_ids[torch.randint(
                    0, action_ids.numel(), (n_action,), device=self.device
                )])
        n_random = batch_size - sum(group.numel() for group in selected)
        selected.insert(0, torch.randint(
            0, self.size, (n_random,), device=self.device))
        ids = torch.cat(selected)
        return (
            self.obs[ids].float() / 255.0,
            self.actions[ids],
            self.rewards[ids],
            self.next_obs[ids].float() / 255.0,
            self.dones[ids],
            self.disc[ids],
        )


SELF_REPLAY_ANCHOR_FORMAT = "dino_rl.image_self_replay_anchor.v1"


def validate_self_replay_anchor(payload, state_shape=None):
    """Validate the protected self-replay file contract."""
    if not isinstance(payload, dict):
        raise ValueError("self replay anchor must contain a dictionary")
    if payload.get("format") != SELF_REPLAY_ANCHOR_FORMAT:
        raise ValueError(
            f"self replay anchor format must be {SELF_REPLAY_ANCHOR_FORMAT!r}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("self replay anchor is missing provenance metadata")
    if provenance.get("kind") != "self_generated_replay":
        raise ValueError("self replay anchor provenance is not self-generated")
    if provenance.get("teacher_or_demo_data") is not False:
        raise ValueError(
            "self replay anchor must explicitly declare no teacher/demo data"
        )

    fields = ("obs", "next_obs", "actions", "rewards", "dones", "disc")
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(f"self replay anchor missing fields: {', '.join(missing)}")
    size = int(payload.get("size", -1))
    if size < 1:
        raise ValueError("self replay anchor size must be positive")
    for name in fields:
        shape = tuple(payload[name].shape)
        if not shape or shape[0] != size:
            raise ValueError(
                f"self replay anchor field {name!r} must have {size} rows"
            )
    if tuple(payload["actions"].shape) != (size,):
        raise ValueError("self replay anchor actions must be one-dimensional")
    for name in ("rewards", "dones", "disc"):
        if tuple(payload[name].shape) != (size,):
            raise ValueError(f"self replay anchor {name} must be one-dimensional")
    if state_shape is not None:
        expected = tuple(state_shape)
        for name in ("obs", "next_obs"):
            if tuple(payload[name].shape[1:]) != expected:
                raise ValueError(
                    f"self replay anchor {name} shape must be (N, {expected})"
                )
    return size, dict(provenance)


def load_self_replay_anchor(path, state_shape, device, chunk_size=2048):
    """Load a CPU-staged torch anchor into an immutable GPU replay buffer."""
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (RuntimeError, ValueError) as exc:
        if "mmap" not in str(exc).lower():
            raise
        payload = torch.load(path, map_location="cpu", weights_only=True)
    size, provenance = validate_self_replay_anchor(payload, state_shape)
    anchor = GPUReplayBuffer(size, state_shape, device)
    dtypes = {
        "obs": torch.uint8,
        "next_obs": torch.uint8,
        "actions": torch.long,
        "rewards": torch.float32,
        "dones": torch.float32,
        "disc": torch.float32,
    }
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        for name, dtype in dtypes.items():
            source = torch.as_tensor(payload[name][start:stop], dtype=dtype)
            getattr(anchor, name)[start:stop].copy_(source)
    anchor.idx = 0
    anchor.size = size
    del payload
    return anchor, provenance


class NStepAccumulator:
    """Turns 1-step transitions into n-step transitions, per parallel env.

    n-step returns propagate reward (esp. the crash penalty) back up to n steps
    in one update, which dramatically speeds credit assignment -- the bottleneck
    for the image agent, which must learn 'obstacle approaching -> danger' before
    it can ever survive. Fully general (no game knowledge). Emits transitions
    carrying their own bootstrap discount gamma^k so truncated/terminal windows
    are handled exactly.
    """

    def __init__(self, num_envs, n, gamma):
        from collections import deque
        self.n = n
        self.gamma = gamma
        self.bufs = [deque() for _ in range(num_envs)]

    def push(
        self, obs, actions, rewards, next_obs, dones, interruptions=None
    ):
        o, a, R, no, d, disc = [], [], [], [], [], []
        interruptions = (
            np.zeros(len(actions), dtype=bool)
            if interruptions is None
            else np.asarray(interruptions, dtype=bool)
        )
        if interruptions.shape != (len(actions),):
            raise ValueError("interruptions must align with the environment batch")

        def emit(i, terminal):
            buf = self.bufs[i]
            ret = 0.0
            for k, (_, _, r) in enumerate(buf):
                ret += (self.gamma ** k) * r
            o.append(buf[0][0]); a.append(buf[0][1]); R.append(ret)
            no.append(next_obs[i]); d.append(1.0 if terminal else 0.0)
            disc.append(self.gamma ** len(buf))

        for i in range(len(actions)):
            if interruptions[i]:
                self.bufs[i].clear()
                continue
            self.bufs[i].append((obs[i], int(actions[i]), float(rewards[i])))
            if dones[i]:
                while self.bufs[i]:           # flush whole episode (truncated returns)
                    emit(i, terminal=True)
                    self.bufs[i].popleft()
            elif len(self.bufs[i]) == self.n:
                emit(i, terminal=False)
                self.bufs[i].popleft()

        if not a:
            return None
        return (np.asarray(o), np.asarray(a, dtype=np.int64),
                np.asarray(R, dtype=np.float32), np.asarray(no),
                np.asarray(d, dtype=np.float32), np.asarray(disc, dtype=np.float32))


def eval_rank(result):
    """Rank evaluations by the lower tail required by the completion gate."""
    return (int(result.get("min", -1)), float(result.get("avg", -1.0)))


def far_jump_penalty_mask(actions, infos, distance_pixels):
    """Select jumps initiated from exact grounded states with a far obstacle."""
    if not np.isfinite(distance_pixels) or distance_pixels < 0:
        raise ValueError("far_jump_distance must be finite and non-negative")
    actions = np.asarray(actions)
    if actions.ndim != 1 or len(actions) != len(infos):
        raise ValueError("actions and infos must be aligned one-dimensional rows")

    mask = np.zeros(len(actions), dtype=bool)
    for i, (action, info) in enumerate(zip(actions, infos)):
        if action != 1 or not isinstance(info, dict):
            continue
        try:
            features = np.asarray(
                info.get("pre_action_features"), dtype=np.float64
            )
        except (TypeError, ValueError):
            continue
        if features.shape != (FEATURE_DIM,) or not np.all(np.isfinite(features)):
            continue
        grounded_noduck = (
            features[3] == 0.0
            and features[4] == 0.0
            and features[5] == 0.0
            and features[9] == 0.0
        )
        visible_obstacle = features[1] > 0.0 and features[2] > 0.0
        far_enough = features[0] * GAME_WIDTH >= distance_pixels
        mask[i] = grounded_noduck and visible_obstacle and far_enough
    return mask


def apply_far_jump_penalty(
    rewards, actions, infos, penalty, distance_pixels
):
    """Subtract an opt-in penalty from eligible pre-action jump transitions."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("far_jump_penalty must be finite and non-negative")
    rewards = np.asarray(rewards)
    if rewards.ndim != 1 or len(rewards) != len(actions):
        raise ValueError("rewards and actions must be aligned one-dimensional rows")
    if penalty == 0.0:
        return rewards
    adjusted = rewards.copy()
    adjusted[far_jump_penalty_mask(actions, infos, distance_pixels)] -= penalty
    return adjusted


def far_jump_reward_metadata(penalty, distance_pixels):
    """Build persisted configuration and provenance for reward-only features."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("far_jump_penalty must be finite and non-negative")
    if not np.isfinite(distance_pixels) or distance_pixels < 0:
        raise ValueError("far_jump_distance must be finite and non-negative")
    return {
        "far_jump_penalty": float(penalty),
        "far_jump_distance": float(distance_pixels),
        "far_jump_reward_provenance": {
            "kind": "pure_image_pre_action_feature_reward_shaping",
            "feature_info_key": "pre_action_features",
            "policy_observation": "pixels_only",
            "game_width_pixels": float(GAME_WIDTH),
        },
    }


def demo_bank_metadata(archive, path):
    """Read safe JSON provenance from a demo-bank npz, with legacy fallback."""
    if "metadata_json" not in archive.files:
        return {
            "format": "legacy_unversioned_image_demo_bank",
            "path": path,
            "teacher_or_demo_data": True,
        }
    raw = archive["metadata_json"]
    try:
        metadata = json.loads(str(raw.item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid demo-bank metadata in {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"demo-bank metadata in {path} must be an object")
    metadata = dict(metadata)
    metadata.setdefault("path", path)
    metadata["teacher_or_demo_data"] = True
    return metadata


def image_checkpoint_metadata(
    args,
    state_shape,
    demo_provenance=None,
    self_provenance=None,
):
    """Build the policy contract and research provenance saved with a model."""
    return {
        "encoder": args.encoder,
        "action_names": IMAGE_ACTION_NAMES,
        "action_mode": "noop_jump" if args.no_duck else "noop_jump_duck",
        "no_duck": bool(args.no_duck),
        "no_duck_margin": float(args.no_duck_margin),
        "deterministic_training": bool(args.deterministic),
        "frames_per_action": int(args.frames_per_action),
        "mask_score": False,
        "training_config": {
            **vars(args),
            "state_shape": tuple(state_shape),
        },
        "provenance": {
            "kind": "browser_image_dqn_training",
            "policy_observation": "pixels_only",
            "teacher_or_demo_data": demo_provenance is not None,
            "init_from": args.init_from or None,
            "demo_bank": demo_provenance,
            "self_replay_anchor": self_provenance,
        },
    }


def validate_resume_request(
    resume_live, buffer_file, init_from, path_exists=os.path.isfile
):
    """Reject resume requests that would silently start a different run."""
    if not resume_live:
        return
    if not buffer_file:
        raise ValueError("--resume-live requires --buffer-file")
    if init_from:
        raise ValueError("--resume-live and --init-from are mutually exclusive")
    if not path_exists(buffer_file):
        raise ValueError(
            f"--resume-live snapshot does not exist: {buffer_file}"
        )


def validate_replay_snapshot(snapshot, capacity, state_shape, require_live):
    """Validate replay tensors and optional live learner state before copying."""
    required = {
        "obs", "next_obs", "actions", "rewards", "dones", "disc",
        "idx", "size",
    }
    if require_live:
        required.update({"online_sd", "target_sd", "opt_sd"})
    missing = required.difference(snapshot)
    if missing:
        raise ValueError(f"replay snapshot is missing: {sorted(missing)}")

    expected_obs_shape = (capacity, *state_shape)
    for name in ("obs", "next_obs"):
        if tuple(snapshot[name].shape) != expected_obs_shape:
            raise ValueError(
                f"replay snapshot {name} shape must be {expected_obs_shape}"
            )
    for name in ("actions", "rewards", "dones", "disc"):
        if tuple(snapshot[name].shape) != (capacity,):
            raise ValueError(
                f"replay snapshot {name} shape must be {(capacity,)}"
            )
    size = int(snapshot["size"])
    index = int(snapshot["idx"])
    if not 0 <= size <= capacity:
        raise ValueError("replay snapshot size is outside its capacity")
    if not 0 <= index < capacity:
        raise ValueError("replay snapshot index is outside its capacity")


def replay_batch_sizes(batch_size, demo_enabled, demo_frac,
                       self_enabled, self_frac):
    """Return demo, protected-self, and live row counts for one TD batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (demo_enabled
            and (not np.isfinite(demo_frac) or not 0.0 <= demo_frac < 1.0)):
        raise ValueError("demo_frac must be in [0, 1)")
    if not np.isfinite(self_frac) or not 0.0 <= self_frac < 1.0:
        raise ValueError("self_frac must be in [0, 1)")
    demo_count = max(int(batch_size * demo_frac), 1) if demo_enabled else 0
    self_count = max(int(batch_size * self_frac), 1) if self_enabled else 0
    live_count = batch_size - demo_count - self_count
    if live_count < 1:
        raise ValueError("demo_frac and self_frac must leave at least one live row")
    return demo_count, self_count, live_count


def adjusted_live_terminal_frac(batch_size, live_count, terminal_frac):
    """Raise the live fraction so the full-batch terminal quota is unchanged."""
    if (not np.isfinite(terminal_frac)
            or not 0.0 <= terminal_frac <= 1.0):
        raise ValueError("terminal_frac must be in [0, 1]")
    if live_count < 1 or live_count > batch_size:
        raise ValueError("live_count must be in [1, batch_size]")
    target_count = int(batch_size * terminal_frac)
    if target_count > live_count:
        raise ValueError(
            "protected replay fractions leave too few live rows to preserve "
            "the terminal quota"
        )
    if target_count == 0:
        return 0.0
    if live_count == batch_size:
        return float(terminal_frac)
    # ReplayBuffer floors live_count * terminal_frac. The half-row offset keeps
    # that integer count exact despite floating-point division.
    return min((target_count + 0.5) / live_count, 1.0)


def dqfd_margin_loss(q_values, actions, demo_count, margin):
    """Compute DQfD margin loss on the demo prefix, excluding all other rows."""
    if demo_count < 1 or demo_count > len(actions):
        raise ValueError("demo_count must select a non-empty action prefix")
    q_demo = q_values[:demo_count]
    action_demo = actions[:demo_count]
    loss_matrix = torch.full_like(q_demo, margin)
    loss_matrix.scatter_(1, action_demo.unsqueeze(1), 0.0)
    return (
        (q_demo + loss_matrix).max(dim=1).values
        - q_demo.gather(1, action_demo.unsqueeze(1)).squeeze(1)
    ).mean()


def build_explorer_mask(num_envs, explore_env_frac):
    """Choose a fixed cohort of vector workers that may take random actions."""
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if (not np.isfinite(explore_env_frac)
            or not 0.0 <= explore_env_frac <= 1.0):
        raise ValueError("explore_env_frac must be in [0, 1]")
    count = int(np.floor(num_envs * explore_env_frac + 0.5))
    mask = np.zeros(num_envs, dtype=bool)
    mask[:count] = True
    return mask


def select_epsilon_greedy_actions(greedy, epsilon, explorer_mask,
                                  random_jump_prob=-1.0, rng=None,
                                  no_duck=False):
    """Apply epsilon-greedy exploration to a fixed subset of vector workers.

    A negative random_jump_prob samples uniformly over the configured action
    set. Non-negative values sample only noop/jump, with the value giving
    the probability of jump.
    """
    if (not np.isfinite(random_jump_prob)
            or not -1.0 <= random_jump_prob <= 1.0):
        raise ValueError("random_jump_prob must be in [-1, 1]")
    greedy = np.asarray(greedy)
    explorer_mask = np.asarray(explorer_mask, dtype=bool)
    if greedy.ndim != 1 or explorer_mask.shape != greedy.shape:
        raise ValueError("greedy actions and explorer mask must be 1-D and aligned")
    if no_duck:
        # The model path is constrained separately, but keep this helper safe
        # for heterogeneous cohorts and externally supplied greedy actions.
        greedy = np.where(greedy == 2, 0, greedy)

    rng = np.random if rng is None else rng
    if random_jump_prob < 0.0:
        action_count = 2 if no_duck else 3
        if hasattr(rng, "integers"):
            random_actions = rng.integers(0, action_count, size=len(greedy))
        else:
            random_actions = rng.randint(0, action_count, size=len(greedy))
    else:
        random_values = (rng.rand(len(greedy)) if hasattr(rng, "rand")
                         else rng.random(len(greedy)))
        random_actions = (random_values < random_jump_prob).astype(np.int64)
    explore_values = (rng.rand(len(greedy)) if hasattr(rng, "rand")
                      else rng.random(len(greedy)))
    explore = (explore_values < epsilon) & explorer_mask
    return np.where(explore, random_actions, greedy)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--encoder", default="impala", choices=["impala", "nature"])
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--image-width", type=int, default=336)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--buffer", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--train-freq", type=int, default=1, help="grad steps per env tick")
    p.add_argument("--min-replay", type=int, default=5000)
    p.add_argument("--terminal-frac", type=float, default=0.0,
                   help="minimum live-batch fraction sampled from terminal n-step rows")
    p.add_argument("--bank", default="",
                   help="npz of harvested good transitions kept in a PROTECTED "
                        "demo buffer (DQfD: mixed sampling + margin loss); '' disables")
    p.add_argument("--self-bank", default="",
                   help="torch self-replay anchor kept in a protected TD-only "
                        "GPU buffer; use extract_self_image_bank.py; '' disables")
    p.add_argument("--self-frac", type=float, default=0.0,
                   help="fraction of every SmoothL1 TD batch drawn from "
                        "--self-bank; 0 disables")
    p.add_argument("--margin", type=float, default=0.8,
                   help="DQfD large-margin for the demo classification loss")
    p.add_argument("--margin-weight", type=float, default=1.0,
                   help="weight of the DQfD margin loss on demo samples")
    p.add_argument("--demo-jump-frac", type=float, default=0.0,
                   help="minimum demo-batch fraction sampled from jump rows; "
                        "0 keeps uniform protected-demo sampling")
    p.add_argument("--init-from", default="",
                   help="checkpoint to warm-start online+target nets from")
    p.add_argument("--reset-advantage", action="store_true",
                   help="reinitialize only the dueling action head after warm-start")
    p.add_argument("--no-duck", action="store_true",
                   help="restrict policy to noop/jump and project raw duck Q below noop")
    p.add_argument("--no-duck-margin", type=float, default=0.01,
                   help="raw Q margin maintained between noop and duck when --no-duck")
    p.add_argument("--speed-curriculum", type=float, default=0.0,
                   help="fraction of TRAINING episodes that start at a random "
                        "speed in [8,13] instead of 6. Fixes the data-starvation "
                        "at max speed (reaching one speed-13 sample normally "
                        "costs 7000+ frames). Eval always starts normally")
    p.add_argument("--resume-live", action="store_true",
                   help="restore online/target/optimizer from the buffer file's "
                        "live snapshot (true continuation) instead of --init-from")
    p.add_argument("--milestone-bonus", type=float, default=0.0,
                   help="+bonus each time an env's score crosses a 100-point "
                        "milestone. Sharpens the advantage the same way the "
                        "feature trainer's clear-bonus did (score-derived, "
                        "observation-agnostic). 0 disables")
    p.add_argument("--clear-bonus", type=float, default=0.0,
                   help="+bonus when the nearest obstacle is passed; the event "
                        "shapes reward but is not part of the image observation")
    p.add_argument("--far-jump-penalty", type=float, default=0.0,
                   help="penalty for a grounded jump with a visible obstacle at "
                        "least --far-jump-distance pixels away; 0 disables")
    p.add_argument("--far-jump-distance", type=float, default=200.0,
                   help="minimum pre-action obstacle distance in pixels for the "
                        "opt-in far-jump penalty")
    p.add_argument("--buffer-file", default="",
                   help="torch file to persist the GPU replay buffer across "
                        "runs (loaded at start if present, saved at each eval)")
    p.add_argument("--demo-frac", type=float, default=0.25,
                   help="fraction of each batch drawn from the protected demo "
                        "buffer (DQfD mixed sampling)")
    p.add_argument("--anneal-after", type=float, default=0,
                   help="once the best eval minimum exceeds this, decay lr 0.6x "
                        "per strictly better lower-tail result (floor 1e-5); 0 "
                        "disables. Stops TD churn from flipping a good policy")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=150000)
    p.add_argument("--explore-env-frac", type=float, default=1.0,
                   help="fixed fraction of vector envs eligible for epsilon "
                        "exploration; remaining envs always act greedily")
    p.add_argument("--random-jump-prob", type=float, default=-1.0,
                   help="when >=0, random exploration is noop/jump only with "
                        "this jump probability; negative keeps uniform 3-action "
                        "exploration")
    p.add_argument("--crash-penalty", type=float, default=-10.0)
    p.add_argument("--alive-reward", type=float, default=0.01)
    p.add_argument("--cpu-buffer", action="store_true",
                   help="Use the slow CPU replay buffer (default: GPU-resident)")
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision grad steps (faster on the B200)")
    p.add_argument("--n-step", type=int, default=1,
                   help="n-step returns (1 = standard DQN); speeds credit assignment")
    p.add_argument("--deterministic", action="store_true",
                   help="Pause-and-frame-step browser env (discrete, synced, faster)")
    p.add_argument("--frames-per-action", type=int, default=2,
                   help="Game frames advanced per env step in deterministic mode")
    p.add_argument("--time-budget-sec", type=float, default=21600)
    p.add_argument("--eval-every-sec", type=float, default=900)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--run-tag", default="dqn")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    try:
        explorer_mask = build_explorer_mask(args.num_envs, args.explore_env_frac)
        if (not np.isfinite(args.random_jump_prob)
                or not -1.0 <= args.random_jump_prob <= 1.0):
            raise ValueError("random_jump_prob must be in [-1, 1]")
        if not np.isfinite(args.no_duck_margin) or args.no_duck_margin <= 0:
            raise ValueError("no_duck_margin must be positive")
        far_jump_metadata = far_jump_reward_metadata(
            args.far_jump_penalty, args.far_jump_distance
        )
        if bool(args.self_bank) != bool(args.self_frac > 0.0):
            raise ValueError(
                "--self-bank and a positive --self-frac must be set together"
            )
        batch_mix = replay_batch_sizes(
            args.batch_size, bool(args.bank), args.demo_frac,
            bool(args.self_bank), args.self_frac,
        )
        demo_count, self_count, live_count = batch_mix
        if args.self_bank:
            live_terminal_frac = adjusted_live_terminal_frac(
                args.batch_size, live_count, args.terminal_frac
            )
        else:
            live_terminal_frac = args.terminal_frac
        validate_resume_request(
            args.resume_live, args.buffer_file, args.init_from
        )
    except ValueError as exc:
        p.error(str(exc))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_shape = (4, args.image_size, args.image_width)
    online = DuelingCNNQ(state_shape, 3, args.encoder).to(device)
    target = DuelingCNNQ(state_shape, 3, args.encoder).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    if args.init_from:
        _ck = torch.load(args.init_from, map_location=device, weights_only=True)
        _sd = (_ck.get("model_state_dict", _ck.get("model", _ck))
               if isinstance(_ck, dict) else _ck)
        online.load_state_dict(_sd)
        target.load_state_dict(_sd)
        print(f"warm-started from {args.init_from}", flush=True)
    if args.reset_advantage:
        online.advantage.reset_parameters()
        target.load_state_dict(online.state_dict())
        print("reinitialized dueling advantage head", flush=True)
    if args.no_duck:
        project_no_duck_advantage(online, args.no_duck_margin)
        project_no_duck_advantage(target, args.no_duck_margin)
        print(f"constrained actions to noop/jump (raw duck margin "
              f"{args.no_duck_margin:g})", flush=True)
    opt = optim.Adam(online.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    buf = (ReplayBuffer(args.buffer, state_shape) if args.cpu_buffer
           else GPUReplayBuffer(args.buffer, state_shape, device))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    n_params = sum(p.numel() for p in online.parameters())
    print(f"Image DQN | encoder={args.encoder} params={n_params/1e6:.2f}M "
          f"obs={state_shape} envs={args.num_envs} device={device} "
          f"buffer={'CPU' if args.cpu_buffer else 'GPU'} amp={args.amp}", flush=True)
    random_mode = (("uniform-2" if args.no_duck else "uniform-3")
                   if args.random_jump_prob < 0.0
                   else f"noop/jump(p={args.random_jump_prob:.3f})")
    print(f"exploration collectors={int(explorer_mask.sum())}/{args.num_envs} "
          f"random_actions={random_mode} no_duck={args.no_duck}", flush=True)
    print(
        "far-jump reward shaping "
        f"penalty={args.far_jump_penalty:g} "
        f"distance={args.far_jump_distance:g}px "
        "source=info.pre_action_features policy_observation=pixels_only",
        flush=True,
    )

    # Optionally seed the buffer from a harvested good-trajectory bank
    # (warm-start / DQfD-lite): pre-fills replay with competent teacher play so
    # early sampling isn't all random-exploration junk -- the exploration wall is
    # why pixel RL plateaus low here. Chunked to avoid a large float temp. Bank
    # obs are uint8 (env-obs*255); /255 back to [0,1] matches how add_batch
    # re-quantizes live transitions, so seeded rows are stored identically.
    # disc=gamma is the 1-step bootstrap discount (terminals masked by done).
    # DQfD-style: the bank lives in a PROTECTED demo buffer that is never
    # evicted (a plain pre-fill decayed as soon as the circular buffer cycled
    # past it: eval 183 -> 80). Each update samples a fixed demo fraction and
    # applies a large-margin classification loss on the demo actions so the
    # policy stays anchored to the teacher while its own experience grows.
    demo_buf = None
    demo_bank_provenance = None
    if args.bank:
        d = np.load(args.bank)
        demo_bank_provenance = demo_bank_metadata(d, args.bank)
        bo, ba, br, bno, bd = (d["obs"], d["actions"], d["rewards"],
                               d["next_obs"], d["dones"])
        if "expert" in d.files:
            keep = d["expert"].astype(bool)
            if not np.all(keep):
                bo, ba, br, bno, bd = (
                    bo[keep], ba[keep], br[keep], bno[keep], bd[keep]
                )
        m = len(ba)
        demo_buf = GPUReplayBuffer(m, state_shape, device)
        for s0 in range(0, m, 4096):
            s1 = min(s0 + 4096, m)
            disc = np.full(s1 - s0, args.gamma, dtype=np.float32)
            demo_buf.add_batch(bo[s0:s1].astype(np.float32) / 255.0, ba[s0:s1],
                               br[s0:s1], bno[s0:s1].astype(np.float32) / 255.0,
                               bd[s0:s1], disc)
        print(f"protected demo buffer from {args.bank}: {m} transitions "
              f"(duck={(ba == 2).sum()})", flush=True)
        d.close()

    self_buf = None
    self_bank_provenance = None
    if args.self_bank:
        self_buf, self_bank_provenance = load_self_replay_anchor(
            args.self_bank, state_shape, device
        )
        print(
            f"protected self replay anchor from {args.self_bank}: "
            f"{self_buf.size} transitions (TD-only, no demo margin)\n"
            f"  provenance={self_bank_provenance!r}",
            flush=True,
        )
        print(
            f"replay batch mix: demo={demo_count} self={self_count} "
            f"live={live_count} live_terminal_frac={live_terminal_frac:.6f}",
            flush=True,
        )
    self_replay_metadata = {
        "self_bank": args.self_bank or None,
        "self_frac": args.self_frac,
        "self_bank_size": self_buf.size if self_buf is not None else 0,
        "self_bank_provenance": self_bank_provenance,
    }
    checkpoint_metadata = image_checkpoint_metadata(
        args,
        state_shape,
        demo_provenance=demo_bank_provenance,
        self_provenance=self_bank_provenance,
    )

    if args.buffer_file:
        import os as _os
        if _os.path.isfile(args.buffer_file):
            # The destination replay is already allocated on the learner
            # device. Stage snapshots in host memory so a resume does not
            # transiently require two full GPU replay buffers.
            _bd = torch.load(
                args.buffer_file, map_location="cpu", weights_only=True
            )
            validate_replay_snapshot(
                _bd, args.buffer, state_shape, require_live=args.resume_live
            )
            buf.obs.copy_(_bd["obs"]); buf.next_obs.copy_(_bd["next_obs"])
            buf.actions.copy_(_bd["actions"]); buf.rewards.copy_(_bd["rewards"])
            buf.dones.copy_(_bd["dones"]); buf.disc.copy_(_bd["disc"])
            buf.idx = int(_bd["idx"]); buf.size = int(_bd["size"])
            print(f"loaded replay buffer: size={buf.size} from {args.buffer_file}",
                  flush=True)
            if args.resume_live:
                online.load_state_dict(_bd["online_sd"])
                target.load_state_dict(_bd["target_sd"])
                opt.load_state_dict(_bd["opt_sd"])
                if args.no_duck:
                    project_no_duck_advantage(online, args.no_duck_margin)
                    project_no_duck_advantage(target, args.no_duck_margin)
                # Optimizer snapshots include their old param-group LR. The CLI
                # LR is the experiment contract and must remain authoritative on
                # a resume (especially for explicit low-LR consolidation runs).
                for group in opt.param_groups:
                    group["lr"] = args.lr
                print("resumed LIVE online/target/optimizer state (true "
                      f"continuation, lr reset to {args.lr:.1e})", flush=True)
            del _bd

    env = VecChromeDinoImageEnv(
        args.num_envs, obs_size=args.image_size, obs_width=args.image_width,
        action_repeat=args.action_repeat, reward_mode="survival",
        mask_score=False,
        deterministic=args.deterministic, frames_per_action=args.frames_per_action,
        start_speed_prob=args.speed_curriculum,
    )
    obs = env.reset()
    N = args.num_envs
    nstep = NStepAccumulator(N, args.n_step, args.gamma)
    milestones = np.zeros(N, dtype=np.int64)  # floor(score/100) per env

    def epsilon(step):
        frac = min(1.0, step / args.eps_decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    @torch.no_grad()
    def greedy_actions(obs_np):
        q = online(torch.as_tensor(obs_np, dtype=torch.float32, device=device))
        return greedy_action_indices(q, args.no_duck).cpu().numpy()

    def evaluate_batch(n_episodes, max_steps):
        eval_env = VecChromeDinoImageEnv(
            n_episodes, obs_size=args.image_size, obs_width=args.image_width,
            reward_mode="survival", deterministic=args.deterministic,
            frames_per_action=args.frames_per_action, mask_score=False,
        )
        try:
            eval_obs = eval_env.reset()
            finished = np.zeros(n_episodes, dtype=bool)
            step_counts = np.zeros(n_episodes, dtype=np.int64)
            recovery_counts = np.zeros(n_episodes, dtype=np.int64)
            scores = np.zeros(n_episodes, dtype=np.int64)
            while not np.all(finished):
                eval_actions = greedy_actions(eval_obs)
                eval_actions[finished] = 0
                eval_obs, _rewards, dones, infos = eval_env.step(eval_actions)
                for i in range(n_episodes):
                    if finished[i]:
                        continue
                    if infos[i].get("browser_recovered", False):
                        recovery_counts[i] += 1
                        if recovery_counts[i] >= 3:
                            raise RuntimeError(f"eval env {i} recovered three times")
                        step_counts[i] = 0
                        scores[i] = 0
                        continue
                    step_counts[i] += 1
                    scores[i] = int(infos[i].get("score", scores[i]))
                    finished[i] = bool(dones[i]) or step_counts[i] >= max_steps
            score_list = scores.tolist()
            return {"avg": float(scores.mean()), "min": int(scores.min()),
                    "max": int(scores.max()), "scores": score_list}
        finally:
            eval_env.close()

    # Initialize best_eval from the existing best checkpoint so a restarted run
    # can never clobber a stronger all-time best with its own weaker first save
    # (this bug silently overwrote the 1405-era nets across restarts).
    best_eval = -1.0
    best_min = -1
    _best_path = f"checkpoints/dino_dqn_image_{args.run_tag}_best.pth"
    try:
        import os as _os
        if _os.path.isfile(_best_path):
            _prev = torch.load(_best_path, map_location="cpu", weights_only=True)
            _prev_res = _prev.get("eval_result") or {}
            _prev_avg = float(_prev_res.get("avg", -1.0))
            _prev_min = int(_prev_res.get("min", -1))
            if (_prev_min, _prev_avg) > (best_min, best_eval):
                best_min, best_eval = _prev_min, _prev_avg
                print("best eval initialized from existing ckpt: "
                      f"min={best_min} avg={best_eval:.1f}", flush=True)
        del _prev
    except Exception:
        pass
    env_steps = 0
    grad_steps = 0
    start = time.time()
    last_eval = start
    recent_scores = []

    while time.time() - start < args.time_budget_sec:
        eps = epsilon(env_steps)
        greedy = greedy_actions(obs)
        actions = select_epsilon_greedy_actions(
            greedy, eps, explorer_mask,
            random_jump_prob=args.random_jump_prob,
            no_duck=args.no_duck,
        )

        next_obs, _env_reward, dones, infos = env.step(actions)
        recoveries = np.asarray([
            bool(info.get("browser_recovered", False)) for info in infos
        ])
        # Reward shaping (matches the feature DQN): bounded survival signal.
        rewards = np.where(dones, args.crash_penalty, args.alive_reward).astype(np.float32)
        rewards = apply_far_jump_penalty(
            rewards,
            actions,
            infos,
            args.far_jump_penalty,
            args.far_jump_distance,
        )
        if args.clear_bonus:
            rewards += np.asarray([
                args.clear_bonus if info.get("obstacle_cleared", False) else 0.0
                for info in infos
            ], dtype=np.float32)
        if args.milestone_bonus:
            for i in range(N):
                if dones[i]:
                    milestones[i] = 0
                    continue
                m = int(infos[i].get("score", 0)) // 100
                if m > milestones[i]:
                    rewards[i] += args.milestone_bonus * (m - milestones[i])
                    milestones[i] = m
        # Accumulate n-step transitions before storing; emits ready transitions
        # (with their bootstrap discount gamma^k) once each env's window fills.
        ready = nstep.push(
            obs, actions, rewards, next_obs, dones,
            interruptions=recoveries,
        )
        if ready is not None:
            o, a, R, no, dn, disc = ready
            buf.add_batch(o, a, R, no, dn, disc)
        obs = next_obs
        env_steps += N
        for i in range(N):
            if dones[i] and not recoveries[i]:
                recent_scores.append(int(infos[i]["score"]))

        # ---- learn ----
        ready = (buf.size >= args.min_replay if demo_buf is None
                 else buf.size >= 2000)
        if ready:
            for _ in range(args.train_freq):
                replay_parts = []
                if demo_buf is not None:
                    replay_parts.append(demo_buf.sample(
                        demo_count, device, action=1,
                        action_frac=args.demo_jump_frac
                    ))
                if self_buf is not None:
                    # The anchor participates in TD regression only. Keeping it
                    # after the demo prefix also excludes it from DQfD margin.
                    replay_parts.append(self_buf.sample(self_count, device))
                replay_parts.append(buf.sample(
                    live_count, device,
                    terminal_frac=live_terminal_frac
                ))
                if len(replay_parts) == 1:
                    s, a, r, ns, d, disc = replay_parts[0]
                else:
                    s, a, r, ns, d, disc = (
                        torch.cat(rows, dim=0)
                        for rows in zip(*replay_parts)
                    )
                with torch.autocast("cuda", enabled=args.amp):
                    qall = online(s)
                    q = qall.gather(1, a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        best_a = greedy_action_indices(
                            online(ns), args.no_duck
                        )                                         # Double DQN
                        next_q = target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
                        # disc is gamma^n (n-step bootstrap discount) per transition
                        tgt = r + (1 - d) * disc * next_q
                    loss = loss_fn(q, tgt)
                    if demo_buf is not None:
                        # Only the leading demo rows receive classification
                        # supervision; protected self/live rows remain TD-only.
                        je = dqfd_margin_loss(
                            qall, a, demo_count, args.margin
                        )
                        loss = loss + args.margin_weight * je
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                scaler.step(opt); scaler.update()
                if args.no_duck:
                    project_no_duck_advantage(online, args.no_duck_margin)
                grad_steps += 1
                for tp, sp in zip(target.parameters(), online.parameters()):
                    tp.data.mul_(1 - args.tau).add_(args.tau * sp.data)
                if args.no_duck:
                    project_no_duck_advantage(target, args.no_duck_margin)

        # ---- periodic log + eval ----
        if time.time() - last_eval >= args.eval_every_sec and buf.size >= args.min_replay:
            sps = env_steps / (time.time() - start)
            tr = np.mean(recent_scores[-50:]) if recent_scores else 0.0
            print(f"[{int(time.time()-start)}s] env_steps={env_steps} grad={grad_steps} "
                  f"eps={eps:.3f} buf={buf.size} steps/s={sps:.1f} "
                  f"train_score~{tr:.0f} loss={loss.item():.4f}", flush=True)
            try:
                res = evaluate_batch(args.eval_episodes, 20000)
                print(f"  >> EVAL avg={res['avg']:.1f} min={res['min']} "
                      f"max={res['max']}", flush=True)
                if res["min"] >= 5000:
                    if args.no_duck:
                        project_no_duck_advantage(online, args.no_duck_margin)
                    save_checkpoint(
                        f"checkpoints/dino_dqn_image_{args.run_tag}_candidate.pth",
                        online, update=grad_steps, eval_result=res,
                        env_backend="browser", observation_mode="image",
                        state_shape=state_shape,
                        **checkpoint_metadata,
                        **far_jump_metadata,
                        **self_replay_metadata,
                    )
                    print("  >> saved formal-verification candidate", flush=True)
                # The formal gate is a minimum score, so checkpoint selection
                # must optimize the lower tail rather than the mean of a small,
                # noisy suite. Strict ranking also prevents repeated all-cap ties
                # from ratcheting the LR down and overwriting the only best.
                if eval_rank(res) > (best_min, best_eval):
                    best_min, best_eval = eval_rank(res)
                    if args.anneal_after and best_min > args.anneal_after:
                        for g in opt.param_groups:
                            g["lr"] = max(g["lr"] * 0.6, 1e-5)
                        print(f"  >> lr annealed to {opt.param_groups[0]['lr']:.1e}",
                              flush=True)
                    if args.no_duck:
                        project_no_duck_advantage(online, args.no_duck_margin)
                    save_checkpoint(
                        f"checkpoints/dino_dqn_image_{args.run_tag}_best.pth",
                        online, update=grad_steps,
                        eval_result=res, env_backend="browser",
                        observation_mode="image", state_shape=state_shape,
                        **checkpoint_metadata,
                        **far_jump_metadata,
                        **self_replay_metadata,
                    )
                    print(f"  >> saved best (min={best_min} avg={best_eval:.1f})",
                          flush=True)
            except Exception as exc:  # eval browser hiccup shouldn't kill training
                print(f"  >> eval skipped: {type(exc).__name__}: {exc}", flush=True)
            if args.buffer_file and not args.cpu_buffer:
                import os as _os
                torch.save({"obs": buf.obs, "next_obs": buf.next_obs,
                            "actions": buf.actions, "rewards": buf.rewards,
                            "dones": buf.dones, "disc": buf.disc,
                            "idx": buf.idx, "size": buf.size,
                            "online_sd": online.state_dict(),
                            "target_sd": target.state_dict(),
                            "no_duck": args.no_duck,
                            "no_duck_margin": args.no_duck_margin,
                            "opt_sd": opt.state_dict(),
                            **checkpoint_metadata,
                            "live_replay_provenance": {
                                "kind": "self_generated_live_replay",
                                "teacher_or_demo_data": False,
                            },
                            **far_jump_metadata,
                            **self_replay_metadata},
                           args.buffer_file + ".tmp")
                _os.replace(args.buffer_file + ".tmp", args.buffer_file)
            last_eval = time.time()

    print(f"DONE env_steps={env_steps} grad_steps={grad_steps} "
          f"best_min={best_min} best_eval={best_eval:.1f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
