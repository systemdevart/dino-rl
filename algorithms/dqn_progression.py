"""
DQN Stability Pack -- Progressive Improvements
================================================
Phase 8: From Vanilla DQN to Dueling Double DQN with Prioritized Replay

This file demonstrates the evolution of deep Q-learning through four
incremental improvements.  Each variant builds on the previous one, so
you can see exactly what changes and why.

References
----------
- Mnih et al., 2015: "Human-level control through deep reinforcement learning"
  (DQN with experience replay and target network)
- van Hasselt et al., 2016: "Deep Reinforcement Learning with Double Q-learning"
  (Double DQN -- decouple action selection from evaluation)
- Wang et al., 2016: "Dueling Network Architectures for Deep Reinforcement Learning"
  (Dueling architecture -- separate value and advantage streams)
- Schaul et al., 2016: "Prioritized Experience Replay"
  (PER -- sample important transitions more often)

Roadmap
-------
Each section adds exactly ONE improvement to the previous agent.  The file
is structured so you can diff any two consecutive classes and see a minimal,
self-contained change.

    Section 1: Vanilla DQN ........... experience replay + target network
    Section 2: + Double DQN .......... decouple selection from evaluation
    Section 3: + Dueling Architecture  separate V(s) and A(s,a) streams
    Section 4: + Prioritized Replay .. sample proportional to |TD error|

The progression addresses distinct failure modes:

    Vanilla DQN   -->  overestimates Q-values because max is biased
    + Double DQN  -->  architecture treats all actions equally even when
                       only the state matters (e.g. far from obstacles)
    + Dueling     -->  uniform sampling wastes time on boring transitions
    + PER         -->  full stability pack
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    DinoFeatureEnv,
    evaluate,
    plot_training,
    save_results,
    create_writer,
    FEATURE_DIM,
    ACTION_SIZE,
)

# ---------------------------------------------------------------------------
# Hyperparameters (shared across all variants for fair comparison)
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.0003       # Adam learning rate
BATCH_SIZE = 256             # Mini-batch size for replay sampling
GAMMA = 0.99                 # Discount factor
TARGET_UPDATE = 1000         # Hard-copy online -> target every N steps
BUFFER_SIZE = 100_000        # Maximum replay buffer capacity
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_END = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.995        # Multiplicative decay per episode
PER_ALPHA = 0.6              # PER: priority exponent (0 = uniform, 1 = full priority)
PER_BETA_START = 0.4         # PER: initial importance-sampling exponent
PER_BETA_END = 1.0           # PER: final IS exponent (annealed over training)

# Device detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================================================================
# Section 1: Vanilla DQN
# ===========================================================================
#
# The original DQN (Mnih et al., 2015) introduced two key ideas that made
# deep Q-learning stable:
#
# 1. EXPERIENCE REPLAY: Instead of learning from consecutive (s, a, r, s')
#    tuples (which are highly correlated), we store transitions in a buffer
#    and sample random mini-batches.  This breaks temporal correlations and
#    lets each transition be reused many times.
#
# 2. TARGET NETWORK: We maintain a separate "target" copy of the Q-network
#    that is only updated periodically (every TARGET_UPDATE steps).  The
#    Bellman target  y = r + gamma * max_a' Q_target(s', a')  uses this
#    frozen copy, which prevents the moving-target problem where both the
#    prediction and the target shift on every gradient step.
#
# Before DQN: Naive deep Q-learning diverged because (a) correlated samples
# caused the network to overfit to recent experience and forget older lessons,
# and (b) the bootstrapped target  r + gamma * max Q(s', a')  chased a moving
# target, creating feedback loops that amplified errors.
#
# After DQN: Replay breaks correlation; the frozen target network gives a
# stable regression target.  Together they make deep Q-learning converge
# reliably on Atari-scale problems.
# ===========================================================================


class QNetwork(nn.Module):
    """
    Standard MLP Q-network: state -> Q(s, a) for each action.

    Architecture:
        state (FEATURE_DIM)
            -> Linear(256) -> ReLU
            -> Linear(256) -> ReLU
            -> Linear(ACTION_SIZE)    <-- Q-values, one per action
    """

    def __init__(self, state_dim: int = FEATURE_DIM, action_dim: int = ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions given state batch."""
        return self.net(x)


class ReplayBuffer:
    """
    Simple experience replay buffer using a deque.

    Stores (state, action, reward, next_state, done) tuples and supports
    uniform random sampling of mini-batches.

    The deque automatically evicts the oldest transition once maxlen is
    reached, giving a sliding window over recent experience.
    """

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Return a random batch as separate numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class VanillaDQN:
    """
    Vanilla DQN agent (Mnih et al., 2015).

    Training procedure for each step:
        1. Select action via epsilon-greedy.
        2. Store transition (s, a, r, s', done) in replay buffer.
        3. Sample a random mini-batch from the buffer.
        4. Compute Bellman target:  y = r + gamma * max_a' Q_target(s', a')
        5. Minimize MSE loss:  L = (Q_online(s, a) - y)^2
        6. Every TARGET_UPDATE steps, hard-copy online -> target network.
    """

    name = "VanillaDQN"

    def __init__(self, device=DEVICE):
        self.device = device
        self.online_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon, pick a random action (explore).
        Otherwise, pick the action with the highest Q-value (exploit).
        """
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def compute_td_targets(self, rewards_t, next_states_t, dones_t):
        """
        Compute Bellman targets: y = r + gamma * max_a' Q_target(s', a').

        This is the vanilla DQN target.  The max is taken over the target
        network's Q-values, which means the same network both selects AND
        evaluates the best next action.  This causes overestimation bias
        (addressed by Double DQN in Section 2).
        """
        with torch.no_grad():
            # max_a' Q_target(s', a')
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            # y = r + gamma * max Q(s', a') * (1 - done)
            targets = rewards_t + GAMMA * next_q * (1.0 - dones_t)
        return targets

    def update(self):
        """
        Sample a batch from replay and perform one gradient step.

        Returns the mean loss value for logging, or None if the buffer
        does not have enough samples yet.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values: Q_online(s, a) -- gather the Q-value for the taken action
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman targets (overridden in Double DQN)
        targets = self.compute_td_targets(rewards_t, next_states_t, dones_t)

        # MSE loss and gradient step
        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard target update
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Multiplicatively decay epsilon after each episode."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def policy(self, state: np.ndarray) -> int:
        """Deterministic greedy policy for evaluation."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()


# ===========================================================================
# Section 2: + Double DQN
# ===========================================================================
#
# Problem with Vanilla DQN:
#   The target  y = r + gamma * max_a' Q_target(s', a')  uses the SAME
#   network to both SELECT the best next action and EVALUATE its value.
#   In noisy environments, this systematically overestimates Q-values
#   because  E[max(X1, X2)] >= max(E[X1], E[X2]).  Over many updates,
#   these overestimates compound and can destabilize learning.
#
# Double DQN fix (van Hasselt et al., 2016):
#   Decouple action selection from evaluation using TWO networks:
#
#       a* = argmax_a Q_ONLINE(s', a)     <-- online network SELECTS
#       y  = r + gamma * Q_TARGET(s', a*) <-- target network EVALUATES
#
#   The online network picks which action looks best, but the target
#   network provides the Q-value for that action.  Since the two networks
#   have different parameters (the target is a stale copy), their errors
#   are less correlated, which dramatically reduces overestimation.
#
# Before (Vanilla DQN): Q-value estimates are systematically inflated;
# the agent thinks bad states are better than they are, leading to
# overconfident and suboptimal policies.
#
# After (Double DQN): Overestimation is greatly reduced.  The agent
# learns more accurate Q-values and typically achieves better scores
# with the same number of training steps.
#
# Implementation change: ONLY the compute_td_targets method changes.
# Everything else (network, buffer, training loop) stays identical.
# ===========================================================================


class DoubleDQN(VanillaDQN):
    """
    Double DQN agent (van Hasselt et al., 2016).

    Inherits everything from VanillaDQN; only overrides the target
    computation to decouple action selection from evaluation.
    """

    name = "DoubleDQN"

    def compute_td_targets(self, rewards_t, next_states_t, dones_t):
        """
        Double DQN target:
            a* = argmax_a Q_online(s', a)       <-- select with online
            y  = r + gamma * Q_target(s', a*)   <-- evaluate with target

        Contrast with Vanilla DQN which does both with the target network:
            y  = r + gamma * max_a Q_target(s', a)
        """
        with torch.no_grad():
            # Step 1: Online network selects the best action for each next state
            best_actions = self.online_net(next_states_t).argmax(dim=1)

            # Step 2: Target network evaluates Q-value of that action
            next_q = self.target_net(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            # y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
            targets = rewards_t + GAMMA * next_q * (1.0 - dones_t)
        return targets


# ===========================================================================
# Section 3: + Dueling Architecture
# ===========================================================================
#
# Observation (Wang et al., 2016):
#   In many states, the VALUE of being in that state matters more than the
#   specific action you take.  For example, when the dino is far from any
#   obstacle, Q(s, jump) and Q(s, do_nothing) are nearly identical -- the
#   state itself is "safe" regardless of what you do.  A standard Q-network
#   has to learn this redundancy for every action independently.
#
# Dueling Architecture:
#   Split the Q-network into two streams after shared feature layers:
#
#       Shared features ──┬──> V(s)       (scalar: how good is this state?)
#                         └──> A(s, a)    (per-action: advantage of each action)
#
#   Recombine as:
#       Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))
#
#   The mean subtraction makes V and A identifiable (otherwise you could
#   shift a constant between them without changing Q).  Now V(s) only needs
#   to learn one number per state, and A(s, a) only needs to capture the
#   relative benefit of each action.  This is more sample-efficient.
#
# Before (standard Q-network): The network must independently learn accurate
# Q-values for every (state, action) pair, even when actions have similar
# values.
#
# After (Dueling): V(s) can generalize across actions; updates to V(s)
# immediately improve Q-values for ALL actions in that state.  The agent
# learns faster, especially in states where action choice doesn't matter
# much (which is most states in the Dino game).
#
# Implementation change: Replace QNetwork with DuelingQNetwork.  The
# training algorithm (Double DQN targets) is unchanged.
# ===========================================================================


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network (Wang et al., 2016).

    Architecture:
        state (FEATURE_DIM)
            -> Linear(256) -> ReLU
            -> Linear(256) -> ReLU          <-- shared feature layers
            ├── Linear(256) -> ReLU -> Linear(1)            --> V(s)
            └── Linear(256) -> ReLU -> Linear(ACTION_SIZE)  --> A(s, a)

        Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))
    """

    def __init__(self, state_dim: int = FEATURE_DIM, action_dim: int = ACTION_SIZE):
        super().__init__()

        # Shared feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Value stream: V(s) -- one scalar per state
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream: A(s, a) -- one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a) = V(s) + A(s, a) - mean(A(s, .)).

        The mean subtraction ensures identifiability: without it, we could
        add a constant c to V(s) and subtract c from every A(s, a) without
        changing Q.  The mean-centering forces the advantage to have zero
        mean, so V(s) unambiguously captures the state value.
        """
        features = self.features(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DuelingDoubleDQN(DoubleDQN):
    """
    Dueling Double DQN agent: combines Dueling architecture with Double DQN.

    Inherits Double DQN target computation; replaces the network architecture
    with DuelingQNetwork.
    """

    name = "DuelingDoubleDQN"

    def __init__(self, device=DEVICE):
        # Skip DoubleDQN.__init__ and VanillaDQN.__init__ to use DuelingQNetwork
        # instead of QNetwork.  We replicate the setup here with the dueling variant.
        self.device = device
        self.online_net = DuelingQNetwork().to(device)
        self.target_net = DuelingQNetwork().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.step_count = 0


# ===========================================================================
# Section 4: + Prioritized Experience Replay (PER)
# ===========================================================================
#
# Problem with uniform replay:
#   Vanilla experience replay samples transitions uniformly at random.  This
#   means boring, well-learned transitions (dino running in the open with no
#   obstacles) are sampled just as often as rare, surprising transitions
#   (barely missing a cactus cluster, or discovering a new pterodactyl
#   pattern).  The agent wastes most of its gradient updates re-learning
#   things it already knows.
#
# Prioritized Experience Replay (Schaul et al., 2016):
#   Sample transitions PROPORTIONALLY to their TD error:
#
#       priority_i = |delta_i|^alpha + epsilon_per
#
#   where delta_i = r + gamma * Q_target(s', a*) - Q_online(s, a) is the
#   temporal-difference error.  Large |delta| means the network was
#   surprised by this transition -- it predicted badly and has the most
#   to learn from it.
#
#   alpha controls how much prioritization matters:
#       alpha = 0  -->  uniform replay (no prioritization)
#       alpha = 1  -->  full prioritization (greedy)
#
# Importance sampling correction:
#   Prioritized sampling introduces bias: transitions with high priority are
#   seen more often, so the gradient estimate is no longer an unbiased sample
#   of the full buffer.  We correct this with importance-sampling weights:
#
#       w_i = (1 / (N * P(i)))^beta
#
#   where P(i) is the sampling probability and N is the buffer size.
#   beta is annealed from BETA_START to 1.0 over training.  At beta=1.0,
#   the weights fully correct the bias; at beta=0 they have no effect.
#   We anneal rather than using beta=1 immediately because early in training,
#   we care more about learning speed than unbiased gradients.
#
# Data structure -- Sum Tree:
#   Naive priority sampling requires O(N) to compute the CDF.  A sum tree
#   (segment tree) stores priorities at the leaves and partial sums at
#   internal nodes, enabling O(log N) sampling and O(log N) priority updates.
#
# Before (uniform replay): The agent samples boring transitions as often as
# informative ones.  Rare but critical failure cases (crashing into obstacles)
# may be underrepresented.
#
# After (PER): The agent focuses learning on transitions it finds surprising,
# dramatically improving sample efficiency.  Combined with the other three
# improvements, this gives us the "full stability pack" for DQN.
# ===========================================================================


class SumTree:
    """
    Binary sum tree for O(log n) prioritized sampling.

    Structure:
        - Leaves hold individual priorities (one per stored transition).
        - Each internal node holds the sum of its children.
        - The root holds the total sum of all priorities.

    Sampling:
        To sample proportionally, draw a uniform random number in [0, total],
        then traverse the tree top-down: go left if the random number is less
        than the left child's sum, otherwise subtract the left sum and go
        right.  This runs in O(log n) time.

    Array layout:
        Index 0 is the root.  For node at index i:
            left child  = 2*i + 1
            right child = 2*i + 2
        Leaves start at index (capacity - 1).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_index = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Update parent sums after a leaf priority changes."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _leaf_index(self, idx: int) -> int:
        """Convert data index to tree leaf index."""
        return idx + self.capacity - 1

    def update(self, data_idx: int, priority: float):
        """Set the priority for the transition at data_idx."""
        tree_idx = self._leaf_index(data_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if tree_idx > 0:
            self._propagate(tree_idx, change)

    def add(self, priority: float, data):
        """Add a transition with the given priority."""
        self.data[self.write_index] = data
        self.update(self.write_index, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, value: float) -> int:
        """
        Walk down the tree to find the leaf corresponding to `value`.

        Starting at node `idx`, go left if value < left child's sum,
        otherwise subtract the left sum and go right.
        """
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left >= len(self.tree):
            return idx  # Reached a leaf

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def sample(self, value: float):
        """
        Sample one transition: given a uniform value in [0, total], return
        (tree_index, priority, data).
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - (self.capacity - 1)
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def total(self) -> float:
        """Total priority sum (root of tree)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Maximum priority among all stored leaves."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.size
        if self.size == 0:
            return 1.0
        return max(self.tree[leaf_start:leaf_end])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using a sum tree.

    New transitions are added with maximum priority (so they are guaranteed
    to be sampled at least once).  After training on a batch, priorities
    are updated to reflect the latest TD errors.

    Sampling:
        1. Divide [0, total_priority] into batch_size equal segments.
        2. Sample one value uniformly from each segment (stratified sampling).
        3. Walk the sum tree to find the corresponding transition.
        4. Compute importance-sampling weights to correct the bias.
    """

    # Small constant added to priorities to ensure no transition has zero
    # probability of being sampled.
    EPSILON_PER = 1e-6

    def __init__(self, capacity: int = BUFFER_SIZE, alpha: float = PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition with maximum priority.

        New transitions get the highest priority in the buffer so that
        they are guaranteed to be sampled at least once.  Their priority
        will be corrected after the first training update.
        """
        max_p = self.tree.max_priority
        if max_p == 0:
            max_p = 1.0
        data = (state, action, reward, next_state, done)
        self.tree.add(max_p, data)

    def sample(self, batch_size: int, beta: float):
        """
        Sample a prioritized batch.

        Args:
            batch_size: number of transitions to sample
            beta: importance-sampling exponent (0 = no correction, 1 = full)

        Returns:
            states, actions, rewards, next_states, dones: batch arrays
            indices: tree indices (needed to update priorities later)
            weights: importance-sampling weights (used to scale the loss)
        """
        indices = []
        priorities = []
        batch = []
        total = self.tree.total

        # Stratified sampling: divide [0, total] into batch_size segments
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            tree_idx, priority, data = self.tree.sample(value)
            indices.append(tree_idx)
            priorities.append(priority)
            batch.append(data)

        # Compute importance-sampling weights
        #   P(i) = priority_i / total_priority
        #   w_i  = (1 / (N * P(i)))^beta
        # Normalize by max weight so that weights only scale downward
        # (this avoids exploding gradients from very large weights).
        n = self.tree.size
        priorities = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities / total
        weights = (n * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize to [0, 1]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on new TD errors.

        priority = (|td_error| + epsilon)^alpha
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.EPSILON_PER) ** self.alpha
            self.tree.update(idx - (self.tree.capacity - 1), priority)

    def __len__(self):
        return self.tree.size


class DuelingDoubleDQN_PER(DuelingDoubleDQN):
    """
    Full stability pack: Dueling Double DQN + Prioritized Experience Replay.

    Inherits the Dueling architecture and Double DQN targets from
    DuelingDoubleDQN.  Replaces uniform replay with prioritized replay
    and modifies the update step to use importance-sampling weights.
    """

    name = "DuelingDoubleDQN_PER"

    def __init__(self, device=DEVICE):
        super().__init__(device)
        # Replace uniform buffer with prioritized buffer
        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE, PER_ALPHA)
        self.beta = PER_BETA_START

    def update(self):
        """
        PER-aware update: sample by priority, weight loss by IS weights.

        The key difference from uniform replay is two-fold:
        1. Sampling: transitions with high |TD error| are sampled more often.
        2. Loss weighting: each sample's loss contribution is scaled by its
           importance-sampling weight to correct the non-uniform sampling bias.
        3. After the gradient step, priorities are updated with new TD errors.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        # Sample prioritized batch
        (states, actions, rewards, next_states, dones,
         indices, weights) = self.buffer.sample(BATCH_SIZE, self.beta)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Current Q-values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN targets (inherited from DoubleDQN)
        targets = self.compute_td_targets(rewards_t, next_states_t, dones_t)

        # TD errors (for priority update)
        td_errors = (current_q - targets).detach().cpu().numpy()

        # Weighted MSE loss: each sample is scaled by its IS weight
        # This corrects the bias introduced by non-uniform sampling.
        elementwise_loss = (current_q - targets) ** 2
        loss = (weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer with new TD errors
        self.buffer.update_priorities(indices, td_errors)

        # Hard target update
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def anneal_beta(self, episode: int, total_episodes: int):
        """
        Linearly anneal beta from PER_BETA_START to PER_BETA_END.

        Early in training (small beta), we prioritize learning speed over
        unbiased gradients.  As training progresses, beta approaches 1.0
        to fully correct the sampling bias, which matters more when the
        policy is nearly converged and we need precise Q-value estimates.
        """
        fraction = min(1.0, episode / total_episodes)
        self.beta = PER_BETA_START + fraction * (PER_BETA_END - PER_BETA_START)


# ===========================================================================
# Training Loop (shared by all variants)
# ===========================================================================

def train_agent(
    agent,
    n_episodes: int = 500,
    print_every: int = 10,
    eval_every: int = 100,
    writer=None,
):
    """
    Train a DQN-family agent on the Dino game.

    This generic loop works for all four variants because they share the
    same interface: select_action, store_transition, update, decay_epsilon,
    and policy.

    Args:
        agent: one of VanillaDQN, DoubleDQN, DuelingDoubleDQN,
               or DuelingDoubleDQN_PER
        n_episodes: number of training episodes
        print_every: print progress every N episodes
        eval_every: run deterministic evaluation every N episodes

    Returns:
        train_scores: list of per-episode scores
        eval_history: list of (episode, avg_score) tuples
    """
    env = DinoFeatureEnv()
    train_scores = []
    eval_history = []

    print(f"\n{'='*60}")
    print(f"Training {agent.name}  |  device={agent.device}  |  {n_episodes} episodes")
    print(f"{'='*60}")

    if writer is None:
        writer = create_writer(f'dqn_{agent.name}')

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            state = next_state

        # Decay exploration
        agent.decay_epsilon()

        # Anneal PER beta if applicable
        if hasattr(agent, 'anneal_beta'):
            agent.anneal_beta(episode, n_episodes)

        score = info['score']
        train_scores.append(score)

        writer.add_scalar('train/score', score, episode)
        if losses:
            writer.add_scalar('train/loss', np.mean(losses), episode)
        writer.add_scalar('train/epsilon', agent.epsilon, episode)

        # Progress reporting
        if episode % print_every == 0:
            recent_avg = np.mean(train_scores[-print_every:])
            avg_loss = np.mean(losses) if losses else 0.0
            print(
                f"  Ep {episode:4d} | "
                f"Score {score:4d} | "
                f"Avg({print_every}) {recent_avg:6.1f} | "
                f"Loss {avg_loss:.4f} | "
                f"Eps {agent.epsilon:.3f}"
            )
            writer.add_scalar('train/avg_score', recent_avg, episode)

        # Periodic evaluation
        if episode % eval_every == 0:
            eval_result = evaluate(agent.policy)
            eval_history.append((episode, eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], episode)
            print(
                f"  >> Eval @ {episode}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

    writer.close()

    return train_scores, eval_history


# ===========================================================================
# Entry Point -- Train all variants and compare
# ===========================================================================

if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    n_episodes = 500

    print(f"Device: {DEVICE}")
    print(f"Training all four DQN variants for {n_episodes} episodes each.\n")

    # Store results for final comparison
    comparison = {}

    # ----- Variant 1: Vanilla DQN -----------------------------------------
    agent1 = VanillaDQN(DEVICE)
    scores1, evals1 = train_agent(agent1, n_episodes=n_episodes)
    final_eval1 = evaluate(agent1.policy)
    comparison['VanillaDQN'] = final_eval1
    save_results('dqn_vanilla', scores1, eval_result=final_eval1)
    plot_training(
        scores1,
        title='Vanilla DQN (Replay + Target Network)',
        path=os.path.join(results_dir, 'dqn_vanilla.png'),
        eval_scores=evals1,
    )

    # ----- Variant 2: Double DQN ------------------------------------------
    agent2 = DoubleDQN(DEVICE)
    scores2, evals2 = train_agent(agent2, n_episodes=n_episodes)
    final_eval2 = evaluate(agent2.policy)
    comparison['DoubleDQN'] = final_eval2
    save_results('dqn_double', scores2, eval_result=final_eval2)
    plot_training(
        scores2,
        title='Double DQN (+ Decoupled Selection/Evaluation)',
        path=os.path.join(results_dir, 'dqn_double.png'),
        eval_scores=evals2,
    )

    # ----- Variant 3: Dueling Double DQN ----------------------------------
    agent3 = DuelingDoubleDQN(DEVICE)
    scores3, evals3 = train_agent(agent3, n_episodes=n_episodes)
    final_eval3 = evaluate(agent3.policy)
    comparison['DuelingDoubleDQN'] = final_eval3
    save_results('dqn_dueling_double', scores3, eval_result=final_eval3)
    plot_training(
        scores3,
        title='Dueling Double DQN (+ V/A Stream Split)',
        path=os.path.join(results_dir, 'dqn_dueling_double.png'),
        eval_scores=evals3,
    )

    # ----- Variant 4: Dueling Double DQN + PER ----------------------------
    agent4 = DuelingDoubleDQN_PER(DEVICE)
    scores4, evals4 = train_agent(agent4, n_episodes=n_episodes)
    final_eval4 = evaluate(agent4.policy)
    comparison['DuelingDoubleDQN_PER'] = final_eval4
    save_results('dqn_dueling_double_per', scores4, eval_result=final_eval4)
    plot_training(
        scores4,
        title='Dueling Double DQN + PER (Full Stability Pack)',
        path=os.path.join(results_dir, 'dqn_dueling_double_per.png'),
        eval_scores=evals4,
    )

    # ----- Final Comparison -----------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (20-episode evaluation)")
    print("=" * 60)
    print(f"{'Algorithm':<28s} {'Avg':>7s} {'Min':>6s} {'Max':>6s}")
    print("-" * 50)
    for name, result in comparison.items():
        print(f"{name:<28s} {result['avg']:7.1f} {result['min']:6d} {result['max']:6d}")
    print("=" * 60)
