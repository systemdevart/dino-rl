"""
DQN Agent for Chrome Dinosaur Game.

Uses Dueling Double DQN with experience replay and soft target updates.
Trains on a pure Python simulation of the Chrome dino game.

Key fixes from original:
- np.argmax -> np.max in Q-learning update (critical bug fix)
- Double DQN for reduced overestimation bias
- Dueling architecture for better value estimation
- Feature-based observation (distance to obstacle, dino height, etc.)
  instead of raw pixels for faster convergence
- Soft target updates (polyak averaging) for stable Q-values
- Batch training on GPU
- Train every N steps during gameplay (not just at episode end)
- Periodic eval episodes with no exploration
- Switched from TensorFlow/Keras to PyTorch for better compatibility
"""
import numpy as np
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from dino_rl.env import DinoRunEnv
from dino_rl.feature_contract import (
    FEATURE_DIM,
    GAME_HEIGHT,
    GAME_WIDTH,
    MAX_GAME_SPEED,
    MAX_OBSTACLE_HEIGHT,
    MAX_OBSTACLE_WIDTH,
)
from dino_rl.networks import DuelingDQN
from dino_rl.policy_paths import DQN_CHECKPOINT_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Opt-in obstacle-cleared reward: +this per obstacle passed. Makes the action
# advantage LARGE (jump->clear vs noop->crash) instead of Q~=V (advantage ~0.01),
# so the learned policy is robust enough to transfer to the browser. 0 = off.
_CLEAR_BONUS = float(os.environ.get("DQN_CLEAR_REWARD", 0.0))


def greedy_feature_actions(
    q_values: torch.Tensor,
    *,
    no_duck: bool = False,
) -> torch.Tensor:
    """Select feature-policy actions while enforcing the configured action set."""
    if q_values.ndim != 2 or q_values.shape[1] < 3:
        raise ValueError("q_values must have shape (batch, at least 3)")
    if no_duck:
        return q_values[:, :2].argmax(dim=1)
    return q_values.argmax(dim=1)


_EMPTY_STATE_ATOL = 1e-4


def grounded_empty_state_mask(states: torch.Tensor) -> torch.Tensor:
    """Identify grounded states where the feature contract has no obstacle."""
    if states.ndim != 2 or states.shape[1] != FEATURE_DIM:
        raise ValueError(f"states must have shape (batch, {FEATURE_DIM})")
    no_distance = (states[:, 0] - 1.0).abs() <= _EMPTY_STATE_ATOL
    no_width = states[:, 1].abs() <= _EMPTY_STATE_ATOL
    no_height = states[:, 2].abs() <= _EMPTY_STATE_ATOL
    grounded = states[:, 5] < 0.5
    return grounded & no_distance & no_width & no_height


def noop_over_jump_hinge_loss(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Apply a noop-over-jump hinge to rows selected by a boolean mask."""
    if q_values.ndim != 2 or q_values.shape[1] < 2:
        raise ValueError("q_values must contain noop and jump actions")
    if mask.ndim != 1 or mask.shape[0] != q_values.shape[0]:
        raise ValueError("mask must select rows from q_values")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    violations = torch.relu(margin + q_values[:, 1] - q_values[:, 0])
    selected = violations[mask]
    if selected.numel() == 0:
        return q_values[:, :2].sum() * 0.0
    return selected.mean()


def empty_state_noop_hinge_loss(
    q_values: torch.Tensor,
    states: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require noop to beat jump by margin in grounded empty states."""
    return noop_over_jump_hinge_loss(
        q_values, grounded_empty_state_mask(states), margin
    )


_GROUNDED_STATE_ATOL = 1e-4


def grounded_visible_far_state_mask(
    states: torch.Tensor, distance_pixels: float
) -> torch.Tensor:
    """Select grounded real-obstacle states at or beyond a pixel distance."""
    if states.ndim != 2 or states.shape[1] != FEATURE_DIM:
        raise ValueError(f"states must have shape (batch, {FEATURE_DIM})")
    if not np.isfinite(distance_pixels) or distance_pixels < 0:
        raise ValueError("distance_pixels must be finite and non-negative")
    near_ground = states[:, 3].abs() <= _GROUNDED_STATE_ATOL
    near_zero_velocity = states[:, 4].abs() <= _GROUNDED_STATE_ATOL
    not_jumping = states[:, 5] < 0.5
    not_ducking = states[:, 9] < 0.5
    visible_obstacle = (states[:, 1] > 0) & (states[:, 2] > 0)
    far_enough = states[:, 0] * GAME_WIDTH >= distance_pixels
    return (
        near_ground
        & near_zero_velocity
        & not_jumping
        & not_ducking
        & visible_obstacle
        & far_enough
    )


def far_state_noop_hinge_loss(
    q_values: torch.Tensor,
    states: torch.Tensor,
    distance_pixels: float,
    margin: float,
) -> torch.Tensor:
    """Require noop to beat jump for grounded visible obstacles still far away."""
    mask = grounded_visible_far_state_mask(states, distance_pixels)
    return noop_over_jump_hinge_loss(q_values, mask, margin)


_SEQUENCE_GEOMETRY_ATOL = 1e-4
_SEQUENCE_SPEED_MIN_PIXELS = 12.5
_SEQUENCE_DIST1_MIN_PIXELS = 150.0
_SEQUENCE_DIST1_MAX_PIXELS = 200.0
_SEQUENCE_DIST2_MIN_PIXELS = 525.0
_SEQUENCE_DIST2_MAX_PIXELS = 590.0


def is_forensic_sequence_jump_state(state) -> bool:
    """Return whether one feature vector matches the forensic sequence window."""
    features = np.asarray(state)
    if features.shape != (FEATURE_DIM,):
        raise ValueError(f"state must have shape ({FEATURE_DIM},)")
    return bool(
        abs(float(features[3])) <= _GROUNDED_STATE_ATOL
        and abs(float(features[4])) <= _GROUNDED_STATE_ATOL
        and float(features[5]) < 0.5
        and float(features[9]) < 0.5
        and float(features[7]) * MAX_GAME_SPEED >= _SEQUENCE_SPEED_MIN_PIXELS
        and np.isclose(
            features[1], 17.0 / MAX_OBSTACLE_WIDTH,
            rtol=0.0, atol=_SEQUENCE_GEOMETRY_ATOL,
        )
        and np.isclose(
            features[2], 35.0 / MAX_OBSTACLE_HEIGHT,
            rtol=0.0, atol=_SEQUENCE_GEOMETRY_ATOL,
        )
        and np.isclose(
            features[8], 105.0 / GAME_HEIGHT,
            rtol=0.0, atol=_SEQUENCE_GEOMETRY_ATOL,
        )
        and _SEQUENCE_DIST1_MIN_PIXELS
        <= float(features[0]) * GAME_WIDTH
        <= _SEQUENCE_DIST1_MAX_PIXELS
        and _SEQUENCE_DIST2_MIN_PIXELS
        <= float(features[6]) * GAME_WIDTH
        <= _SEQUENCE_DIST2_MAX_PIXELS
    )


def forensic_sequence_jump_state_mask(states: torch.Tensor) -> torch.Tensor:
    """Select the high-speed two-obstacle sequence found in crash forensics."""
    if states.ndim != 2 or states.shape[1] != FEATURE_DIM:
        raise ValueError(f"states must have shape (batch, {FEATURE_DIM})")

    near_ground = states[:, 3].abs() <= _GROUNDED_STATE_ATOL
    near_zero_velocity = states[:, 4].abs() <= _GROUNDED_STATE_ATOL
    not_jumping = states[:, 5] < 0.5
    not_ducking = states[:, 9] < 0.5
    fast_enough = (
        states[:, 7] * MAX_GAME_SPEED >= _SEQUENCE_SPEED_MIN_PIXELS
    )

    small_cactus_width = torch.isclose(
        states[:, 1],
        states.new_tensor(17.0 / MAX_OBSTACLE_WIDTH),
        rtol=0.0,
        atol=_SEQUENCE_GEOMETRY_ATOL,
    )
    small_cactus_height = torch.isclose(
        states[:, 2],
        states.new_tensor(35.0 / MAX_OBSTACLE_HEIGHT),
        rtol=0.0,
        atol=_SEQUENCE_GEOMETRY_ATOL,
    )
    small_cactus_y = torch.isclose(
        states[:, 8],
        states.new_tensor(105.0 / GAME_HEIGHT),
        rtol=0.0,
        atol=_SEQUENCE_GEOMETRY_ATOL,
    )

    dist1_pixels = states[:, 0] * GAME_WIDTH
    dist2_pixels = states[:, 6] * GAME_WIDTH
    dist1_in_window = (
        (dist1_pixels >= _SEQUENCE_DIST1_MIN_PIXELS)
        & (dist1_pixels <= _SEQUENCE_DIST1_MAX_PIXELS)
    )
    dist2_in_window = (
        (dist2_pixels >= _SEQUENCE_DIST2_MIN_PIXELS)
        & (dist2_pixels <= _SEQUENCE_DIST2_MAX_PIXELS)
    )
    return (
        near_ground
        & near_zero_velocity
        & not_jumping
        & not_ducking
        & fast_enough
        & small_cactus_width
        & small_cactus_height
        & small_cactus_y
        & dist1_in_window
        & dist2_in_window
    )


def sequence_jump_hinge_loss(
    q_values: torch.Tensor,
    states: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require jump to beat noop in the forensic two-obstacle sequence."""
    if q_values.ndim != 2 or q_values.shape[1] < 2:
        raise ValueError("q_values must contain noop and jump actions")
    if q_values.shape[0] != states.shape[0]:
        raise ValueError("states and q_values must have the same batch size")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    mask = forensic_sequence_jump_state_mask(states)
    violations = torch.relu(margin + q_values[:, 0] - q_values[:, 1])
    selected = violations[mask]
    if selected.numel() == 0:
        return q_values[:, :2].sum() * 0.0
    return selected.mean()


class Agent:
    def __init__(
        self,
        action_size: int,
        continue_training: bool = False,
        *,
        memory_capacity: int = 200000,
        no_duck: bool = False,
        no_duck_margin: float = 0.01,
        empty_state_noop_margin: float = 0.0,
        empty_state_noop_weight: float = 0.0,
        far_state_noop_distance: float = 0.0,
        far_state_noop_margin: float = 0.0,
        far_state_noop_weight: float = 0.0,
        sequence_jump_margin: float = 0.0,
        sequence_jump_weight: float = 0.0,
    ):
        self.weight_backup = os.environ.get("DQN_CKPT", DQN_CHECKPOINT_PATH)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_capacity)
        # Kept in the same chronological order as memory. This makes a
        # terminal sampling quota O(batch), without scanning replay on every
        # learner update. remember() evicts from both deques together.
        self.terminal_memory = deque()
        self.sequence_memory = deque()
        self.no_duck = bool(no_duck)
        self.no_duck_margin = float(no_duck_margin)
        self.empty_state_noop_margin = float(empty_state_noop_margin)
        self.empty_state_noop_weight = float(empty_state_noop_weight)
        self.far_state_noop_distance = float(far_state_noop_distance)
        self.far_state_noop_margin = float(far_state_noop_margin)
        self.far_state_noop_weight = float(far_state_noop_weight)
        self.sequence_jump_margin = float(sequence_jump_margin)
        self.sequence_jump_weight = float(sequence_jump_weight)
        if self.no_duck and self.action_size < 3:
            raise ValueError("no_duck requires a three-action DQN")
        if self.no_duck_margin <= 0:
            raise ValueError("no_duck_margin must be positive")
        if (
            not np.isfinite(self.empty_state_noop_margin)
            or self.empty_state_noop_margin < 0
        ):
            raise ValueError(
                "empty_state_noop_margin must be finite and non-negative"
            )
        if (
            not np.isfinite(self.empty_state_noop_weight)
            or self.empty_state_noop_weight < 0
        ):
            raise ValueError(
                "empty_state_noop_weight must be finite and non-negative"
            )
        if (
            not np.isfinite(self.far_state_noop_distance)
            or self.far_state_noop_distance < 0
        ):
            raise ValueError(
                "far_state_noop_distance must be finite and non-negative"
            )
        if (
            not np.isfinite(self.far_state_noop_margin)
            or self.far_state_noop_margin < 0
        ):
            raise ValueError(
                "far_state_noop_margin must be finite and non-negative"
            )
        if (
            not np.isfinite(self.far_state_noop_weight)
            or self.far_state_noop_weight < 0
        ):
            raise ValueError(
                "far_state_noop_weight must be finite and non-negative"
            )
        if (
            not np.isfinite(self.sequence_jump_margin)
            or self.sequence_jump_margin < 0
        ):
            raise ValueError(
                "sequence_jump_margin must be finite and non-negative"
            )
        if (
            not np.isfinite(self.sequence_jump_weight)
            or self.sequence_jump_weight < 0
        ):
            raise ValueError(
                "sequence_jump_weight must be finite and non-negative"
            )
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.gamma = 0.99
        self.epsilon_decay = 0.99
        # Env-var overrides (default = original behavior) for stability studies.
        self.learning_rate = float(os.environ.get("DQN_LR", 0.0003))
        self.tau = float(os.environ.get("DQN_TAU", 0.005))  # Soft target update rate
        self.train_freq = 4  # Train every 4 steps
        self.min_replay_size = 2000  # Min transitions before training starts

        os.makedirs(os.path.dirname(self.weight_backup), exist_ok=True)

        self.model = DuelingDQN(FEATURE_DIM, action_size).to(device)
        self.target_model = DuelingDQN(FEATURE_DIM, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        if continue_training and os.path.isfile(self.weight_backup):
            checkpoint = torch.load(self.weight_backup, map_location=device,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Loaded weights from {self.weight_backup} (eps={self.epsilon:.4f})")

        self.project_no_duck()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def soft_update_target(self):
        """Polyak averaging: target = tau * model + (1-tau) * target."""
        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        self.project_no_duck()

    @torch.no_grad()
    def project_no_duck(self):
        """Make raw action 2 strictly worse than noop for every input."""
        if not self.no_duck:
            return
        for model in (self.model, self.target_model):
            advantage = model.advantage
            advantage.weight[2].copy_(advantage.weight[0])
            advantage.bias[2].copy_(advantage.bias[0] - self.no_duck_margin)

    def greedy_actions(
        self,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return greedy actions while respecting the configured action set."""
        return greedy_feature_actions(q_values, no_duck=self.no_duck)

    def save_model(self, path: str | None = None):
        self.project_no_duck()
        torch.save({
            'algo': 'dqn',
            'feature_dim': FEATURE_DIM,
            'action_size': self.action_size,
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'no_duck': self.no_duck,
            'no_duck_margin': self.no_duck_margin,
            "empty_state_noop_margin": self.empty_state_noop_margin,
            "empty_state_noop_weight": self.empty_state_noop_weight,
            "far_state_noop_distance": self.far_state_noop_distance,
            "far_state_noop_margin": self.far_state_noop_margin,
            "far_state_noop_weight": self.far_state_noop_weight,
            "sequence_jump_margin": self.sequence_jump_margin,
            "sequence_jump_weight": self.sequence_jump_weight,
        }, path or self.weight_backup)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() <= self.epsilon:
            action_count = 2 if self.no_duck else self.action_size
            return random.randrange(action_count)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_t)
            return self.greedy_actions(q_values).item()

    def remember(self, state, action, reward, next_state, done, discount=None):
        """Append a legacy 1-step row or a discounted n-step replay row."""
        if self.no_duck and int(action) == 2:
            return False
        sequence_state = is_forensic_sequence_jump_state(state)
        if len(self.memory) == self.memory.maxlen:
            evicted = self.memory[0]
            if bool(evicted[4]) and self.terminal_memory:
                self.terminal_memory.popleft()
            if (
                is_forensic_sequence_jump_state(evicted[0])
                and self.sequence_memory
            ):
                self.sequence_memory.popleft()
        transition = (state, action, reward, next_state, done)
        if discount is not None:
            transition += (discount,)
        self.memory.append(transition)
        if bool(done):
            self.terminal_memory.append(transition)
        if sequence_state:
            self.sequence_memory.append(transition)
        return True

    def sample_replay(
        self,
        batch_size: int,
        terminal_frac: float = 0.0,
        sequence_frac: float = 0.0,
    ):
        """Sample replay with disjoint sequence-first and terminal quotas."""
        if not 0.0 <= terminal_frac <= 1.0:
            raise ValueError("terminal_frac must be in [0, 1]")
        if not 0.0 <= sequence_frac <= 1.0:
            raise ValueError("sequence_frac must be in [0, 1]")
        if terminal_frac + sequence_frac > 1.0:
            raise ValueError("terminal_frac + sequence_frac must be at most 1")
        if len(self.memory) < batch_size:
            raise ValueError("replay contains fewer rows than batch_size")

        n_sequence = min(
            int(batch_size * sequence_frac), len(self.sequence_memory)
        )
        minibatch = (
            random.sample(self.sequence_memory, n_sequence)
            if n_sequence else []
        )
        selected_ids = {id(row) for row in minibatch}

        n_terminal = min(
            int(batch_size * terminal_frac), batch_size - len(minibatch)
        )
        if n_terminal:
            # At most len(selected_ids) terminal-cache rows can be excluded, so
            # this draw is large enough to satisfy the quota whenever possible.
            terminal_draw = random.sample(
                self.terminal_memory,
                min(
                    len(self.terminal_memory),
                    n_terminal + len(selected_ids),
                ),
            )
            terminal_rows = [
                row for row in terminal_draw if id(row) not in selected_ids
            ][:n_terminal]
            minibatch.extend(terminal_rows)
            selected_ids.update(id(row) for row in terminal_rows)

        # A size-batch draw contains at most len(selected_ids) reserved rows,
        # so it always contains enough new rows to complete the unique batch.
        general_draw = random.sample(self.memory, batch_size)
        for row in general_draw:
            if id(row) not in selected_ids:
                minibatch.append(row)
                selected_ids.add(id(row))
                if len(minibatch) == batch_size:
                    break
        if len(minibatch) != batch_size:
            raise RuntimeError("replay rows must be distinct transition objects")
        if n_sequence or n_terminal:
            random.shuffle(minibatch)
        return minibatch

    def replay(
        self,
        batch_size: int,
        terminal_frac: float = 0.0,
        sequence_frac: float = 0.0,
    ):
        if len(self.memory) < batch_size:
            return

        minibatch = self.sample_replay(
            batch_size, terminal_frac, sequence_frac
        )

        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch], dtype=np.float32)
        next_states = np.array([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch], dtype=np.float32)
        discounts = np.array([
            s[5] if len(s) >= 6 else self.gamma for s in minibatch
        ], dtype=np.float32)

        states_t = torch.FloatTensor(states).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        discounts_t = torch.FloatTensor(discounts).to(device)

        all_q = self.model(states_t)
        current_q = all_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online model selects action, target model evaluates
            best_actions = self.greedy_actions(self.model(next_states_t))
            next_q = self.target_model(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * discounts_t * next_q

        loss = self.loss_fn(current_q, target_q)
        if (
            self.empty_state_noop_margin > 0
            and self.empty_state_noop_weight > 0
        ):
            empty_noop_loss = empty_state_noop_hinge_loss(
                all_q, states_t, self.empty_state_noop_margin
            )
            loss = loss + self.empty_state_noop_weight * empty_noop_loss
        if (
            self.far_state_noop_distance > 0
            and self.far_state_noop_margin > 0
            and self.far_state_noop_weight > 0
        ):
            far_noop_loss = far_state_noop_hinge_loss(
                all_q,
                states_t,
                self.far_state_noop_distance,
                self.far_state_noop_margin,
            )
            loss = loss + self.far_state_noop_weight * far_noop_loss
        if self.sequence_jump_margin > 0 and self.sequence_jump_weight > 0:
            sequence_jump_loss = sequence_jump_hinge_loss(
                all_q, states_t, self.sequence_jump_margin
            )
            loss = loss + self.sequence_jump_weight * sequence_jump_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.project_no_duck()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TRexRunner:
    """Training loop for the DQN dino agent."""

    def __init__(self, continue_training: bool = False):
        self.batch_size = int(os.environ.get("DQN_BATCH", 256))
        self.episodes = 1000000 # aka "until convergence" - we will stop early if we reach target score
        self.eval_freq = 25  # Run eval episode every N training episodes
        self.eval_runs = int(os.environ.get("DQN_EVAL_RUNS", 5))  # avg over N eval eps
        self.target_score = int(os.environ.get("DQN_TARGET", 20000))
        # Robustness knobs (env vars) -- default OFF preserves old behavior.
        # Training WITH domain randomization + feature noise produces a policy
        # robust to the browser's ~2-8% feature differences (the sim DQN with
        # noise=0 collapses at just 2% input noise -> why it doesn't transfer).
        dr = os.environ.get("DQN_DR", "0") == "1"
        noise = float(os.environ.get("DQN_NOISE", 0.0))
        self.env = DinoRunEnv(domain_randomization=dr, feature_noise=noise,
                              skip_clear_time=True)
        # Eval env uses no randomization for consistent benchmarking
        self.eval_env = DinoRunEnv(domain_randomization=False, feature_noise=0.0,
                                   skip_clear_time=True)
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.action_size, continue_training=continue_training)

    def run_eval_episode(self):
        """Run eval_runs episodes with epsilon=0, return average score."""
        scores = []
        total_steps = 0
        for _ in range(self.eval_runs):
            self.eval_env.reset()
            state = self.eval_env.get_features()
            game_score = 0
            for t in range(25000):
                action = self.agent.act(state, eval_mode=True)
                _, _, score, done = self.eval_env.step(action)
                state = self.eval_env.get_features()
                game_score = score
                if done:
                    break
            scores.append(game_score)
            total_steps += t + 1
        return int(np.mean(scores)), total_steps // self.eval_runs, min(scores)

    def run(self):
        total_time = 0
        best_score = 0
        best_eval = 0
        scores_window = deque(maxlen=100)

        total_params = sum(p.numel() for p in self.agent.model.parameters())
        print(f"Network size: {total_params:,} parameters")

        try:
            for e in range(self.episodes):
                self.env.reset()
                _cleared_seen = set()
                state = self.env.get_features()

                game_score = 0

                for t in range(25000):
                    total_time += 1

                    action = self.agent.act(state)
                    _, reward, score, done = self.env.step(action)

                    # Reward shaping (matching actor-critic's proven config):
                    # +0.01 per step keeps discounted survival reward small,
                    # -10.0 crash penalty dominates for short (bad) episodes.
                    # DQN_CRASH_PENALTY env override for stability studies (a -10
                    # penalty vs +0.01/step creates large TD spikes).
                    if done:
                        reward = float(os.environ.get("DQN_CRASH_PENALTY", -10.0))
                    else:
                        reward = 0.01
                        if _CLEAR_BONUS:
                            for _o in self.env.obstacles:
                                if (_o.x + _o.width <= self.env.dino_x
                                        and id(_o) not in _cleared_seen):
                                    _cleared_seen.add(id(_o))
                                    reward += _CLEAR_BONUS

                    next_state = self.env.get_features()

                    self.agent.remember(state, action, reward, next_state, done)

                    # Train and soft-update target
                    if (total_time % self.agent.train_freq == 0 and
                            len(self.agent.memory) >= self.agent.min_replay_size):
                        self.agent.replay(self.batch_size)
                        self.agent.soft_update_target()

                    state = next_state
                    game_score = score

                    if done:
                        break

                self.agent.decay_epsilon()

                scores_window.append(game_score)
                avg_score = np.mean(scores_window)

                if game_score > best_score:
                    best_score = game_score

                print(
                    f"Ep {e+1:4d}/{self.episodes} | "
                    f"Score: {game_score:5d} | Best: {best_score:5d} | "
                    f"Avg100: {avg_score:7.1f} | "
                    f"Eps: {self.agent.epsilon:.4f} | "
                    f"Steps: {t+1:5d} | Mem: {len(self.agent.memory):6d}"
                )

                # Periodic eval with no exploration
                if (e + 1) % self.eval_freq == 0:
                    eval_score, eval_steps, eval_min = self.run_eval_episode()
                    print(
                        f"  ** EVAL ({self.eval_runs} runs): "
                        f"Avg {eval_score:5d} | Min {eval_min:5d} | "
                        f"Steps {eval_steps:5d} | "
                        f"Best eval: {best_eval:5d}"
                    )
                    if eval_score > best_eval:
                        best_eval = eval_score
                        self.agent.save_model()  # BEST -> weight_backup (never overwritten below)
                        print(f"  ** New best eval! Saved model.")

                    if best_eval >= self.target_score:
                        print(f"\n*** TARGET REACHED! Eval avg: {best_eval} ***")
                        break

                # Periodic/crash-recovery save goes to a SEPARATE '.last' file so it
                # never clobbers the best-eval checkpoint (previously it did, which
                # silently destroyed the good policy once training degraded).
                if (e + 1) % 100 == 0:
                    self.agent.save_model(self.agent.weight_backup + ".last")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print(f"\nFinal save... Best train: {best_score}, Best eval: {best_eval}")
            self.agent.save_model(self.agent.weight_backup + ".last")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN agent for Chrome Dino')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training from saved checkpoint')
    args = parser.parse_args()
    dino = TRexRunner(continue_training=args.continue_training)
    dino.run()


if __name__ == '__main__':
    main()
