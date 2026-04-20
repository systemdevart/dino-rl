"""
PPO -- Proximal Policy Optimization
====================================
Schulman et al., 2017  (arXiv:1707.06347)

PPO is the workhorse algorithm of modern reinforcement learning.  It is used
to train game-playing agents, robotic controllers, and -- most visibly --
large language models via RLHF (Reinforcement Learning from Human Feedback).
Understanding PPO deeply is therefore one of the highest-leverage things you
can do if you are studying RL.

This file implements PPO-Clip with a shared actor-critic network on the
Chrome Dino game.  Every concept is commented in detail so the code doubles
as a tutorial.


Motivation -- why not just use REINFORCE or vanilla policy gradient?
---------------------------------------------------------------------
REINFORCE (Ch 13.3 Sutton & Barto) works, but it has two serious problems:

1.  **High variance.**  Each gradient estimate is based on a *single* Monte
    Carlo rollout.  Lucky episodes push the policy in one direction, unlucky
    episodes push it the other way, and the learning signal is noisy.

2.  **Destructive updates.**  Because there is no constraint on the *size*
    of the policy update, a single bad gradient step can catastrophically
    change the policy.  If the policy suddenly becomes near-deterministic
    in the wrong direction, it collects only bad trajectories from then on
    and may never recover.

The core insight behind PPO (and its predecessor TRPO) is:

    **Large policy updates are dangerous.**
    If we change the policy too much in one step, the new policy may
    visit completely different states, making our collected data (which was
    gathered under the OLD policy) a poor guide for improvement.

This is the *trust region* idea: only update the policy within a "trust
region" where our local approximation of the objective is still accurate.


Trust regions and TRPO (background)
-------------------------------------
TRPO (Schulman et al., 2015) formalised this by imposing a hard KL-divergence
constraint on each update:

    maximise  E[ (pi_new(a|s) / pi_old(a|s)) * A(s,a) ]
    subject to  KL(pi_old || pi_new) <= delta

This requires computing the KL divergence and solving a constrained
optimisation problem (using conjugate gradients + line search), which is
complex to implement and expensive to compute.

PPO achieves *similar* stability with a much simpler mechanism: **clipping**.


The PPO-Clip objective
-----------------------
Instead of a hard KL constraint, PPO clips the objective so that the policy
cannot move too far from the old policy in a single update step.

Define the probability ratio:

    r(theta) = pi_theta(a | s) / pi_old(a | s)

When r(theta) = 1, the new and old policies agree perfectly.

The clipped surrogate objective is:

    L^CLIP(theta) = E[ min( r(theta) * A,
                            clip(r(theta), 1 - epsilon, 1 + epsilon) * A ) ]

where epsilon is a small hyperparameter (typically 0.2).

How does clipping work?
~~~~~~~~~~~~~~~~~~~~~~~
Consider two cases:

**Case 1: Advantage A > 0  (the action was better than expected)**
We want to *increase* the probability of this action, i.e., push r(theta)
above 1.  But the clip(r, 1-eps, 1+eps) term caps the benefit at r = 1+eps.
Beyond that, the clipped term becomes (1+eps)*A which is constant, so the
min() with the unclipped r*A selects the *smaller* value, removing the
incentive to push r further.  The policy is allowed to increase the action's
probability somewhat, but not too aggressively.

**Case 2: Advantage A < 0  (the action was worse than expected)**
We want to *decrease* the probability, i.e., push r(theta) below 1.  But
the clip caps at r = 1-eps.  Below that, the clipped term is (1-eps)*A
(a constant), and the min() again selects the less extreme value, preventing
the policy from *over-correcting* on a bad action.

In both cases, the clipping prevents the policy ratio from deviating too far
from 1, which keeps the new policy close to the old one -- achieving the
trust-region effect without any KL computation.


Why PPO is simpler than TRPO but equally stable
-------------------------------------------------
-  **No second-order optimisation.**  TRPO needs the Fisher information
   matrix (or conjugate gradient approximation) to enforce the KL constraint.
   PPO uses only first-order gradients (standard backpropagation).

-  **No line search.**  TRPO needs a line search to find the largest step
   that satisfies the constraint.  PPO just clips the objective.

-  **Standard SGD / Adam optimiser.**  PPO works with off-the-shelf
   optimisers and standard minibatch training.  TRPO needs a custom
   optimisation routine.

-  **Equally effective.**  Empirically, PPO matches or exceeds TRPO on
   almost all benchmarks (Atari, MuJoCo, etc.), while being much easier
   to implement and tune.  This is why PPO has become the default
   on-policy algorithm in practice.


Generalized Advantage Estimation (GAE)
---------------------------------------
Schulman et al., 2016  (arXiv:1506.02438)

The advantage function A(s, a) = Q(s, a) - V(s) measures how much better
an action is compared to the average action in that state.  Estimating A
well is crucial for low-variance, low-bias policy gradient updates.

There is a fundamental **bias-variance tradeoff** in advantage estimation:

- **1-step TD advantage** (low bias but high variance if V is inaccurate):
      delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  This is the TD error.  It bootstraps off V, introducing bias if V is
  wrong, but has low variance because it only uses one step of actual
  rewards.

- **Monte Carlo advantage** (zero bias but high variance):
      A_t = G_t - V(s_t)    where G_t = sum of discounted future rewards
  This uses no bootstrapping (zero bias) but the full return G_t can be
  very noisy (high variance).

GAE provides a smooth interpolation between these extremes using a parameter
lambda (0 <= lambda <= 1):

    A_t^GAE = sum_{l=0}^{T-t-1}  (gamma * lambda)^l  *  delta_{t+l}

where
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

is the 1-step TD error at time t.

- **lambda = 0:**  A_t = delta_t  (pure 1-step TD -- low variance, high bias)
- **lambda = 1:**  A_t = G_t - V(s_t)  (Monte Carlo -- no bias, high variance)
- **lambda = 0.95** (typical):  a good balance that includes multi-step
  information while still benefiting from the variance reduction of
  bootstrapping.

The GAE computation is analogous to computing returns: we iterate backwards
through the rollout:

    A_{T-1} = delta_{T-1}
    A_t     = delta_t  +  gamma * lambda * A_{t+1}

This is O(T) time and very efficient.


PPO with an Actor-Critic architecture
---------------------------------------
PPO uses a *shared* neural network with two heads:

    state  ->  [shared backbone]  ->  policy head  (actor:  pi(a|s))
                                  ->  value head   (critic: V(s))

Sharing layers is efficient: the backbone learns a good state representation
that both the policy and value function can use.  The two heads are trained
jointly:

    Total loss = policy_loss  +  value_coeff * value_loss  -  entropy_coeff * entropy

where:
    - policy_loss:  the negative clipped surrogate objective (we minimise)
    - value_loss:   (V(s) - returns)^2  (mean squared error)
    - entropy:      -sum pi(a|s) log pi(a|s)  (encourages exploration)

The entropy bonus prevents the policy from becoming deterministic too early,
which would kill exploration.


PPO training loop (high level)
-------------------------------
Unlike REINFORCE (which updates after every episode), PPO collects a fixed
number of steps T ("rollout") and then performs multiple epochs of minibatch
updates on that data.  This is much more sample-efficient because we reuse
each piece of experience K times (typically K=4 epochs).

    for each PPO update:
        1.  Collect T steps using the current policy
            - store: states, actions, rewards, dones, log_probs, values
            - if an episode ends mid-rollout, reset and keep collecting
        2.  Compute GAE advantages and returns for the whole rollout
        3.  For K epochs:
            - shuffle the rollout data
            - split into minibatches
            - for each minibatch:
                a. compute new log_probs and values under current policy
                b. ratio = exp(new_log_prob - old_log_prob)
                c. clipped surrogate loss
                d. value loss
                e. entropy bonus
                f. gradient step
        4.  Discard the rollout data (on-policy: old data is now stale)
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from dino_rl.common import (
    DinoFeatureEnv,
    evaluate,
    plot_training,
    save_results,
    create_writer,
    FEATURE_DIM,
    ACTION_SIZE,
    RESULTS_DIR,
)
from dino_rl.policy_paths import get_ppo_checkpoint_paths

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# We collect them here for easy reference and tuning.
#
# lr           : learning rate for Adam.  3e-4 is the "safe default" for PPO
#                (Andrychowicz et al., 2021 -- "What Matters In On-Policy RL?").
# clip_eps     : the epsilon in clip(r, 1-eps, 1+eps).  0.2 is the standard
#                value from the original PPO paper.
# gamma        : discount factor.  0.99 is standard for most environments.
# lam_gae      : GAE lambda.  0.95 balances bias and variance well.
# ppo_epochs   : how many passes over the rollout buffer per PPO update.
#                4 is the most common choice.  More epochs = more sample
#                reuse, but risk overfitting to stale data.
# rollout_len  : T, number of steps to collect per rollout.  2048 is the
#                standard in MuJoCo benchmarks; for simpler envs it still
#                works well.
# minibatch_sz : size of each minibatch during the PPO update epochs.
#                Must evenly divide rollout_len for clean splitting.
# entropy_coeff: weight of the entropy bonus in the total loss.  Encourages
#                exploration.  0.01 is a common starting point.
# value_coeff  : weight of the value function loss.  0.5 is standard.

LR = 0.0003
CLIP_EPS = 0.2
GAMMA = 0.99
LAM_GAE = 0.95
PPO_EPOCHS = 10
ROLLOUT_LEN = 40000  # Must be > 32,600 frames needed for score 10k
MINIBATCH_SIZE = 512
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
TARGET_EVAL_SCORE = 10000
SCORE_DELTA_COEFF = 0.02
LATENT_DIM = 128

# Browser-backed PPO needs shorter rollouts because every environment step is
# a Selenium round-trip instead of an in-process Python function call.
BROWSER_ROLLOUT_LEN = 4096
BROWSER_PPO_EPOCHS = 4
BROWSER_MINIBATCH_SIZE = 256
BROWSER_EVAL_EPISODES = 5
BROWSER_EVAL_MAX_STEPS = 50000
BROWSER_IMAGE_SIZE = 84
BROWSER_IMAGE_STACK = 4
BROWSER_IMAGE_ACTION_REPEAT = 4
BROWSER_IMAGE_ROLLOUT_LEN = 512
BROWSER_IMAGE_PPO_EPOCHS = 4
BROWSER_IMAGE_MINIBATCH_SIZE = 64
BROWSER_IMAGE_EVAL_EPISODES = 3
BROWSER_IMAGE_EVAL_MAX_STEPS = 6250


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    *,
    update: int | None = None,
    eval_result: dict | None = None,
    env_backend: str = "sim",
    observation_mode: str = "feature",
    state_shape: tuple[int, ...] | None = None,
    score_delta_coeff: float = SCORE_DELTA_COEFF,
):
    """Persist a PPO policy so it can be replayed later in sim or browser."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "algo": "ppo",
        "action_size": ACTION_SIZE,
        "latent_dim": getattr(model, "latent_dim", LATENT_DIM),
        "env_backend": env_backend,
        "observation_mode": observation_mode,
        "model_state_dict": model.state_dict(),
        "score_delta_coeff": score_delta_coeff,
    }
    if hasattr(model, "state_dim"):
        payload["feature_dim"] = getattr(model, "state_dim")
    if state_shape is not None:
        payload["state_shape"] = tuple(state_shape)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if update is not None:
        payload["update"] = update
    if eval_result is not None:
        payload["eval_result"] = eval_result
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    *,
    load_optimizer: bool = False,
):
    """Load PPO weights, optionally restoring the optimizer state too."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_optimizer and optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except ValueError as exc:
                print(f"Skipping optimizer state from {path}: {exc}")
    return checkpoint


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic network for PPO.

    Architecture:
        state (FEATURE_DIM=10)
            -> Linear(128) -> ReLU
            -> Linear(128) -> ReLU          <-- shared backbone
            +--> Linear(ACTION_SIZE=3)      <-- policy head (actor)
            +--> Linear(1)                  <-- value head (critic)

    The shared backbone learns a general state representation.  The two heads
    specialise: the policy head outputs action logits (fed into a softmax via
    Categorical), and the value head outputs a scalar state value V(s).

    Why share layers?
    - Fewer parameters: the backbone is shared, so we learn one good
      representation instead of two separate ones.
    - Implicit regularisation: the value function loss acts as an auxiliary
      task that helps the backbone learn better features, which also benefits
      the policy.
    - This is the standard PPO architecture for small-to-medium problems.
      For very large problems (e.g., Atari with CNN backbones), separate
      networks are sometimes used to avoid interference.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Shared backbone -- three hidden layers with ReLU activation.
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

        # Policy head (actor): outputs raw logits for each action.
        # We do NOT apply softmax here; torch.distributions.Categorical
        # handles log-softmax internally for numerical stability.
        self.policy_head = nn.Linear(latent_dim, action_dim)

        # Value head (critic): outputs a single scalar V(s).
        self.value_head = nn.Linear(latent_dim, 1)

    def encode(self, x: torch.Tensor):
        """Encode a state into a compact latent representation."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x: state tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            logits: action logits of shape (batch, action_dim) or (action_dim,)
            value:  state value of shape (batch, 1) or (1,)
        """
        features = self.encode(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor):
        """
        Given a state, sample an action and return everything PPO needs.

        This is a convenience method used during rollout collection.

        Returns:
            action:   sampled action (int tensor)
            log_prob: log pi(action | state)
            value:    V(state) -- detached scalar
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_action(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Given a batch of states and actions, compute the quantities needed
        for the PPO loss.

        This is used during the PPO update epochs, where we re-evaluate
        actions that were taken under the OLD policy using the CURRENT
        (updated) policy.

        Returns:
            log_probs: log pi_current(actions | states)  -- shape (batch,)
            values:    V_current(states)                  -- shape (batch,)
            entropy:   entropy of pi_current(.|states)    -- shape (batch,)
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class ImageActorCritic(nn.Module):
    """CNN actor-critic that learns a latent embedding from stacked frames."""

    def __init__(
        self,
        state_shape: tuple[int, int, int],
        action_dim: int,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.state_shape = tuple(state_shape)
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        channels, height, width = self.state_shape
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_dim = self.encoder_cnn(dummy).flatten(1).shape[1]

        self.encoder_head = nn.Sequential(
            nn.Linear(conv_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(latent_dim, action_dim)
        self.value_head = nn.Linear(latent_dim, 1)

    def encode(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.encoder_cnn(x)
        features = features.flatten(1)
        return self.encoder_head(features)

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_action(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Fixed-size buffer for storing a single rollout of T steps.

    During collection, we fill the buffer step-by-step.  When an episode ends
    mid-rollout, we simply reset the environment and keep collecting.  The
    buffer stores everything PPO needs:

        states[t]    : the state at step t
        actions[t]   : the action taken at step t
        rewards[t]   : the reward received after taking actions[t]
        dones[t]     : whether the episode ended after step t
        log_probs[t] : log pi_old(actions[t] | states[t])  -- used for ratio
        values[t]    : V_old(states[t])  -- used for GAE computation

    After collection, we compute GAE advantages and returns, then the buffer
    can be iterated over in random minibatches for the PPO update epochs.
    """

    def __init__(
        self,
        rollout_len: int,
        state_shape: tuple[int, ...] | int,
        device: str,
    ):
        self.rollout_len = rollout_len
        self.device = device
        self.ptr = 0  # current write position
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        self.state_shape = tuple(state_shape)

        # Pre-allocate tensors for efficiency.
        self.states = torch.zeros((rollout_len, *self.state_shape), device=device)
        self.next_states = torch.zeros((rollout_len, *self.state_shape), device=device)
        self.actions = torch.zeros(rollout_len, dtype=torch.long, device=device)
        self.rewards = torch.zeros(rollout_len, device=device)
        self.dones = torch.zeros(rollout_len, device=device)
        self.log_probs = torch.zeros(rollout_len, device=device)
        self.values = torch.zeros(rollout_len, device=device)

        # These are computed after the rollout is complete.
        self.advantages = torch.zeros(rollout_len, device=device)
        self.returns = torch.zeros(rollout_len, device=device)

    def store(self, state, next_state, action, reward, done, log_prob, value):
        """Store one transition at the current pointer position."""
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """
        Compute Generalized Advantage Estimation (GAE) for the entire rollout.

        The computation proceeds backwards from the last step:

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t  +  gamma * lambda * (1 - done_t) * A_{t+1}

        The (1 - done_t) factor is crucial: when an episode ends at step t,
        there is no future value to bootstrap from, so V(s_{t+1}) should be
        treated as 0.  Multiplying by (1 - done_t) achieves this.

        After computing advantages, the returns are:
            returns_t = advantages_t + values_t

        This follows from the definition:
            A_t = Q_t - V_t   =>   Q_t = A_t + V_t
        and we use Q_t (the action-value estimate) as the target for V.

        Args:
            last_value: V(s_T), the value of the state *after* the last step
                        in the rollout.  Needed for bootstrapping the final
                        advantage.  If the last step was terminal, this
                        should be 0.
            gamma: discount factor
            lam: GAE lambda
        """
        gae = 0.0
        T = self.rollout_len

        for t in reversed(range(T)):
            # V(s_{t+1}): either the value of the next step in the rollout,
            # or the bootstrapped last_value for the final step.
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            # TD error (1-step advantage estimate):
            #   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )

            # GAE recursive formula:
            #   A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns = advantages + values.
        # These serve as the regression targets for the value function.
        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size: int):
        """
        Yield random minibatch indices for one epoch over the rollout.

        We shuffle the indices and split them into chunks of minibatch_size.
        This ensures each data point is used exactly once per epoch.
        """
        indices = np.arange(self.rollout_len)
        np.random.shuffle(indices)
        for start in range(0, self.rollout_len, minibatch_size):
            end = start + minibatch_size
            batch_idx = indices[start:end]
            yield torch.tensor(batch_idx, dtype=torch.long, device=self.device)

    def reset(self):
        """Reset the write pointer for the next rollout."""
        self.ptr = 0


# ---------------------------------------------------------------------------
# PPO Update
# ---------------------------------------------------------------------------


def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    entropy_coeff: float,
    value_coeff: float,
):
    """
    Perform the PPO parameter update given a filled rollout buffer.

    This function implements the core PPO-Clip algorithm:

    For K epochs:
        For each minibatch in the shuffled rollout:
            1. Re-evaluate stored actions under the CURRENT policy to get
               new log_probs, new values, and entropy.
            2. Compute the probability ratio:
                   ratio = exp(log_prob_new - log_prob_old)
               This is numerically equivalent to pi_new(a|s) / pi_old(a|s)
               but more stable (avoids dividing two small probabilities).
            3. Compute the clipped surrogate objective:
                   surr1 = ratio * advantage
                   surr2 = clip(ratio, 1-eps, 1+eps) * advantage
                   policy_loss = -mean(min(surr1, surr2))
               The negative sign is because we MINIMISE the loss (equivalent
               to MAXIMISING the clipped objective).
            4. Compute the value function loss:
                   value_loss = mean((new_value - return)^2)
               This is a standard regression loss training V(s) to predict
               the empirical returns.
            5. Compute the entropy bonus:
                   entropy_loss = -mean(entropy)
               The negative sign means that subtracting entropy_coeff *
               entropy_loss from the total loss INCREASES entropy (encourages
               exploration).
            6. Total loss = policy_loss + value_coeff * value_loss
                            + entropy_coeff * entropy_loss
               (Note: entropy_loss is negative entropy, so adding
                entropy_coeff * entropy_loss is the same as subtracting
                the entropy bonus.)

    Returns:
        dict with 'policy_loss', 'value_loss', 'entropy' averaged over all
        minibatch updates (for logging purposes).
    """
    # Pre-fetch the normalised advantages.
    # Normalisation (zero mean, unit variance) is a standard practical trick
    # that stabilises training by ensuring the advantages have a consistent
    # scale regardless of the reward magnitudes.
    advantages = buffer.advantages.clone()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_clip_frac = 0.0
    total_approx_kl = 0.0
    total_ratio_mean = 0.0
    total_adv_mean = 0.0
    total_adv_std = 0.0
    n_updates = 0

    # Track raw (un-normalised) advantage stats for logging
    raw_adv_mean = buffer.advantages.mean().item()
    raw_adv_std = buffer.advantages.std().item()

    for _epoch in range(ppo_epochs):
        for batch_idx in buffer.get_minibatches(minibatch_size):
            # ---- Gather minibatch data -----------------------------------
            mb_states = buffer.states[batch_idx]
            mb_actions = buffer.actions[batch_idx]
            mb_old_log_probs = buffer.log_probs[batch_idx]
            mb_advantages = advantages[batch_idx]
            mb_returns = buffer.returns[batch_idx]

            # ---- Re-evaluate actions under current policy ----------------
            new_log_probs, new_values, entropy = model.evaluate_action(
                mb_states, mb_actions
            )

            # ---- Probability ratio --------------------------------------
            # ratio = pi_new(a|s) / pi_old(a|s)
            #       = exp(log pi_new - log pi_old)
            #
            # When the policy hasn't changed, ratio = exp(0) = 1.
            # As the policy diverges from the old one, ratio moves away from 1.
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            # ---- Clipped surrogate objective -----------------------------
            # surr1: the "unclipped" objective -- just the importance-sampled
            #        advantage.
            surr1 = ratio * mb_advantages

            # surr2: the "clipped" objective -- caps the ratio at [1-eps, 1+eps]
            #        so the policy can't move too far in one step.
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages

            # Take the minimum of the two: this is the pessimistic (conservative)
            # estimate.  It removes the incentive for the policy to change too
            # much, implementing the trust region effect.
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # ---- Value function loss -------------------------------------
            # Standard MSE between predicted V(s) and computed returns.
            # Some implementations also clip the value loss, but the original
            # PPO paper found this unnecessary and it's simpler without it.
            value_loss = torch.mean((new_values - mb_returns) ** 2)

            # ---- Entropy bonus -------------------------------------------
            # Entropy = -sum p(a) log p(a).  Higher entropy = more random policy.
            # We SUBTRACT the entropy from the loss (equivalently, we add
            # -entropy_coeff * mean(entropy) to encourage higher entropy).
            entropy_loss = -torch.mean(entropy)

            # ---- Total loss and gradient step ----------------------------
            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping: prevents catastrophically large updates
            # when the loss landscape is steep.  max_norm=0.5 is a common
            # conservative choice for PPO.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # ---- Accumulate metrics for logging --------------------------
            with torch.no_grad():
                # Clip fraction: how often the ratio was clipped.
                # High clip_frac (>0.3) = policy changing too fast.
                # Zero clip_frac = policy not changing at all.
                clipped = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

                # Approx KL divergence between old and new policy.
                # KL ≈ 0.5 * mean((log_ratio)^2)  (second-order approx)
                # If KL > 0.05, policy is diverging too fast.
                approx_kl = (0.5 * (log_ratio**2)).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()  # flip sign for readability
            total_clip_frac += clipped
            total_approx_kl += approx_kl
            total_ratio_mean += ratio.mean().item()
            n_updates += 1

    n = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
        "clip_frac": total_clip_frac / n,
        "approx_kl": total_approx_kl / n,
        "ratio_mean": total_ratio_mean / n,
        "adv_mean": raw_adv_mean,
        "adv_std": raw_adv_std,
        "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
    }


# ---------------------------------------------------------------------------
# Environment / model helpers
# ---------------------------------------------------------------------------


def make_env(
    env_backend: str,
    observation_mode: str,
    env_kwargs: dict | None = None,
):
    """Build the requested PPO environment."""
    env_kwargs = dict(env_kwargs or {})
    if observation_mode == "feature":
        return DinoFeatureEnv(env_backend=env_backend, **env_kwargs)

    if observation_mode != "image":
        raise ValueError(f"Unsupported observation mode '{observation_mode}'.")
    if env_backend != "browser":
        raise ValueError(
            "Image observations are currently only supported for browser backend."
        )

    from dino_rl.browser_env import ChromeDinoImageEnv

    return ChromeDinoImageEnv(
        headless=env_kwargs.get("browser_headless", True),
        accelerate=env_kwargs.get("browser_accelerate", False),
        page_url=env_kwargs.get("browser_url", "chrome://dino"),
        obs_size=env_kwargs.get("image_size", BROWSER_IMAGE_SIZE),
        frame_stack=env_kwargs.get("frame_stack", BROWSER_IMAGE_STACK),
        action_repeat=env_kwargs.get("action_repeat", BROWSER_IMAGE_ACTION_REPEAT),
    )


def make_model(
    observation_mode: str,
    state_shape: tuple[int, ...],
    action_size: int,
) -> nn.Module:
    """Construct the PPO actor-critic for the chosen observation mode."""
    if observation_mode == "feature":
        if len(state_shape) != 1:
            raise ValueError(f"Expected 1D feature state, got shape {state_shape}.")
        return ActorCritic(state_shape[0], action_size)

    if observation_mode == "image":
        if len(state_shape) != 3:
            raise ValueError(
                f"Expected image state shape (C,H,W), got shape {state_shape}."
            )
        return ImageActorCritic(state_shape, action_size)

    raise ValueError(f"Unsupported observation mode '{observation_mode}'.")


def evaluate_policy(
    policy_fn,
    *,
    observation_mode: str,
    n_episodes: int,
    max_steps: int,
    env_backend: str,
    env_kwargs: dict | None = None,
):
    """Evaluate a policy on feature or image observations."""
    if observation_mode == "feature" and env_backend != "browser":
        return evaluate(
            policy_fn,
            n_episodes=n_episodes,
            max_steps=max_steps,
            env_backend=env_backend,
            env_kwargs=env_kwargs,
        )

    env_kwargs = dict(env_kwargs or {})
    max_attempts = 3 if env_backend == "browser" else 1
    last_error = None

    for attempt in range(1, max_attempts + 1):
        env = None
        try:
            env = make_env(env_backend, observation_mode, env_kwargs)
            scores = []
            for _ in range(n_episodes):
                state = env.reset()
                info = {"score": 0}
                for _ in range(max_steps):
                    action = policy_fn(state)
                    state, _reward, done, info = env.step(action)
                    if done:
                        break
                scores.append(info["score"])
            return {
                "avg": float(np.mean(scores)),
                "min": int(np.min(scores)),
                "max": int(np.max(scores)),
                "scores": scores,
            }
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            print(
                f"  >> Eval environment startup failed "
                f"(attempt {attempt}/{max_attempts}): "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(float(attempt))
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    raise RuntimeError(
        "Evaluation failed without raising a concrete error."
    ) from last_error


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_rollout(
    env,
    model: nn.Module,
    buffer: RolloutBuffer,
    state: np.ndarray,
    device: str,
    score_delta_coeff: float,
):
    """
    Collect T steps of experience using the current policy.

    The agent interacts with the environment for exactly T steps.  If an
    episode ends before T steps are collected, the environment is reset and
    collection continues.  This is standard practice in PPO -- rollouts span
    multiple episodes.

    Important: we store the log_probs and values computed by the OLD policy
    (the policy at the start of this rollout).  During the PPO update, we
    will re-compute these under the CURRENT (updated) policy and use the
    ratio between old and new to form the surrogate objective.

    Args:
        env: the environment to collect experience from
        model: the current actor-critic network (used in eval mode)
        buffer: the rollout buffer to fill
        state: the current state of the environment
        device: torch device

    Returns:
        state: the state after the last step (for the next rollout)
        episode_scores: list of scores from completed episodes during this
                        rollout (for logging)
    """
    buffer.reset()
    episode_scores = []
    prev_score = env.get_score()

    for _step in range(buffer.rollout_len):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        # Get action, log_prob, and value from the current policy.
        action, log_prob, value = model.get_action_and_value(state_t)

        # Take the action in the environment.
        next_state, reward, done, info = env.step(action.item())
        score_delta = max(info["score"] - prev_score, 0)
        prev_score = info["score"]
        reward = reward + score_delta_coeff * score_delta
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device)

        # Store the transition.
        buffer.store(
            state=state_t,
            next_state=next_state_t,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        if done:
            # Episode ended.  Record the score and reset.
            episode_scores.append(info["score"])
            state = env.reset()
            prev_score = env.get_score()
        else:
            state = next_state

    # Compute V(s_T) for bootstrapping the final advantage.
    # If the last step was terminal, the GAE computation will zero this out
    # via the (1 - done) factor, but we still need a value for the math.
    last_state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    _, last_value = model(last_state_t)
    last_value = last_value.squeeze(-1).item()

    # Compute GAE advantages and returns.
    buffer.compute_gae(last_value, gamma=GAMMA, lam=LAM_GAE)

    return state, episode_scores


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    n_updates: int = 100,
    device: str | None = None,
    eval_every: int = 10,
    print_every: int = 1,
    writer=None,
    time_budget_sec: float | None = None,
    *,
    env_backend: str = "sim",
    observation_mode: str = "feature",
    env_kwargs: dict | None = None,
    eval_env_backend: str | None = None,
    eval_observation_mode: str | None = None,
    eval_env_kwargs: dict | None = None,
    rollout_len: int | None = None,
    ppo_epochs: int | None = None,
    minibatch_size: int | None = None,
    eval_episodes: int | None = None,
    eval_max_steps: int | None = None,
    init_checkpoint_path: str | None = None,
    load_optimizer_state: bool = False,
    target_eval_score: int = TARGET_EVAL_SCORE,
    algo_name: str | None = None,
    lr: float = LR,
    clip_eps: float = CLIP_EPS,
    entropy_coeff: float = ENTROPY_COEFF,
    value_coeff: float = VALUE_COEFF,
    score_delta_coeff: float = SCORE_DELTA_COEFF,
):
    """
    Train a policy using PPO on the Dino game.

    The outer loop runs for n_updates PPO iterations (or until time_budget_sec
    is exceeded, whichever comes first).  Each iteration:
        1.  Collects a rollout of T steps.
        2.  Computes GAE advantages and returns.
        3.  Performs K epochs of minibatch PPO updates.

    Args:
        n_updates: max number of PPO update iterations
        device: 'cuda' or 'cpu'; auto-detected if None
        eval_every: run deterministic evaluation every N updates
        print_every: print progress every N updates
        time_budget_sec: stop training after this many seconds (None = no limit)
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if eval_env_backend is None:
        eval_env_backend = env_backend
    if eval_observation_mode is None:
        eval_observation_mode = observation_mode
    env_kwargs = dict(env_kwargs or {})
    eval_env_kwargs = dict(eval_env_kwargs or env_kwargs)

    if rollout_len is None:
        if observation_mode == "image":
            rollout_len = BROWSER_IMAGE_ROLLOUT_LEN
        else:
            rollout_len = (
                BROWSER_ROLLOUT_LEN if env_backend == "browser" else ROLLOUT_LEN
            )
    if ppo_epochs is None:
        if observation_mode == "image":
            ppo_epochs = BROWSER_IMAGE_PPO_EPOCHS
        else:
            ppo_epochs = BROWSER_PPO_EPOCHS if env_backend == "browser" else PPO_EPOCHS
    if minibatch_size is None:
        if observation_mode == "image":
            minibatch_size = BROWSER_IMAGE_MINIBATCH_SIZE
        else:
            minibatch_size = (
                BROWSER_MINIBATCH_SIZE if env_backend == "browser" else MINIBATCH_SIZE
            )
    if eval_episodes is None:
        if eval_observation_mode == "image":
            eval_episodes = BROWSER_IMAGE_EVAL_EPISODES
        else:
            eval_episodes = (
                BROWSER_EVAL_EPISODES if eval_env_backend == "browser" else 20
            )
    if eval_max_steps is None:
        if eval_observation_mode == "image":
            eval_max_steps = BROWSER_IMAGE_EVAL_MAX_STEPS
        else:
            eval_max_steps = (
                BROWSER_EVAL_MAX_STEPS if eval_env_backend == "browser" else 50000
            )
    if algo_name is None:
        algo_name = "ppo" if env_backend == "sim" else f"ppo_{env_backend}"
        if observation_mode != "feature":
            algo_name = f"{algo_name}_{observation_mode}"
    if observation_mode == "image" and score_delta_coeff == SCORE_DELTA_COEFF:
        score_delta_coeff = 0.0

    best_ckpt_path, last_ckpt_path = get_ppo_checkpoint_paths(
        env_backend,
        observation_mode,
    )

    print(f"PPO  |  device={device}  lr={lr}  clip_eps={clip_eps}")
    print(
        f"       env_backend={env_backend}  obs_mode={observation_mode}  "
        f"eval_backend={eval_env_backend}  eval_obs_mode={eval_observation_mode}"
    )
    print(f"       gamma={GAMMA}  lam_gae={LAM_GAE}  epochs={ppo_epochs}")
    print(f"       rollout_len={rollout_len}  minibatch={minibatch_size}")
    print(f"       entropy_coeff={entropy_coeff}  value_coeff={value_coeff}")
    print(f"       score_delta_coeff={score_delta_coeff}")
    print(
        f"       total steps = {n_updates} * {rollout_len} = {n_updates * rollout_len}"
    )
    print()

    env = make_env(env_backend, observation_mode, env_kwargs)
    state = env.reset()
    state_shape = tuple(state.shape)
    model = make_model(observation_mode, state_shape, ACTION_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    loaded_checkpoint = None
    if init_checkpoint_path is not None:
        loaded_checkpoint = load_checkpoint(
            init_checkpoint_path,
            model,
            optimizer,
            load_optimizer=load_optimizer_state,
        )
        print(
            f"Loaded PPO checkpoint from {init_checkpoint_path} "
            f"(update={loaded_checkpoint.get('update', 'n/a')}, "
            f"saved_eval={loaded_checkpoint.get('eval_result', {}).get('avg', 'n/a')})"
        )

    # Linear LR annealing: decay from LR to 0 over n_updates
    def lr_lambda(update):
        return 1.0 - update / n_updates

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    buffer = RolloutBuffer(rollout_len, state_shape, device)

    if writer is None:
        writer = create_writer(algo_name)

    # Tracking
    all_episode_scores: list[int] = []
    eval_history: list[tuple[int, float]] = []  # (update_num, avg_score)
    total_steps = 0
    best_eval_avg = float("-inf")
    if loaded_checkpoint is not None:
        best_eval_avg = loaded_checkpoint.get("eval_result", {}).get(
            "avg",
            best_eval_avg,
        )

    train_start = time.time()

    # ------------------------------------------------------------------
    # PPO update loop
    # ------------------------------------------------------------------
    for update in range(1, n_updates + 1):
        # ---- 1. Collect rollout --------------------------------------
        model.eval()
        state, episode_scores = collect_rollout(
            env,
            model,
            buffer,
            state,
            device,
            score_delta_coeff,
        )
        model.train()

        total_steps += rollout_len
        all_episode_scores.extend(episode_scores)

        # ---- 2. PPO parameter update ---------------------------------
        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            clip_eps=clip_eps,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
        )

        # ---- 3. Anneal learning rate ---------------------------------
        scheduler.step()

        # ---- 4. Logging ----------------------------------------------
        if update % print_every == 0:
            # Average score from episodes completed during this rollout.
            if episode_scores:
                avg_score = np.mean(episode_scores)
                n_eps = len(episode_scores)
            else:
                avg_score = 0.0
                n_eps = 0

            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Update {update:4d}/{n_updates} | "
                f"Steps {total_steps:7d} | "
                f"Episodes {n_eps:3d} | "
                f"AvgScore {avg_score:7.1f} | "
                f"PolicyL {metrics['policy_loss']:7.4f} | "
                f"ValueL {metrics['value_loss']:7.4f} | "
                f"Entropy {metrics['entropy']:.4f} | "
                f"KL {metrics['approx_kl']:.5f} | "
                f"Clip {metrics['clip_frac']:.3f}"
            )
            writer.add_scalar("train/policy_loss", metrics["policy_loss"], update)
            writer.add_scalar("train/value_loss", metrics["value_loss"], update)
            writer.add_scalar("train/entropy", metrics["entropy"], update)
            writer.add_scalar("train/avg_score", avg_score, update)
            writer.add_scalar("train/clip_fraction", metrics["clip_frac"], update)
            writer.add_scalar("train/approx_kl", metrics["approx_kl"], update)
            writer.add_scalar("train/ratio_mean", metrics["ratio_mean"], update)
            writer.add_scalar("train/advantage_mean", metrics["adv_mean"], update)
            writer.add_scalar("train/advantage_std", metrics["adv_std"], update)
            writer.add_scalar("train/grad_norm", metrics["grad_norm"], update)
            writer.add_scalar("train/learning_rate", cur_lr, update)
            if episode_scores:
                writer.add_scalar("train/max_score", max(episode_scores), update)
                writer.add_scalar("train/min_score", min(episode_scores), update)

        # ---- 5. Periodic evaluation ----------------------------------
        if update % eval_every == 0:

            @torch.no_grad()
            def policy_fn(s, _model=model, _device=device):
                s_t = torch.as_tensor(s, dtype=torch.float32, device=_device)
                logits, _ = _model(s_t)
                return logits.argmax().item()

            try:
                eval_result = evaluate_policy(
                    policy_fn,
                    observation_mode=eval_observation_mode,
                    n_episodes=eval_episodes,
                    max_steps=eval_max_steps,
                    env_backend=eval_env_backend,
                    env_kwargs=eval_env_kwargs,
                )
            except Exception as exc:
                if eval_env_backend != "browser":
                    raise
                print(
                    f"  >> Eval skipped @ update {update}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            eval_history.append((len(all_episode_scores), eval_result["avg"]))
            writer.add_scalar("eval/avg_score", eval_result["avg"], update)
            print(
                f"  >> Eval @ update {update}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

            if eval_result["avg"] > best_eval_avg:
                best_eval_avg = eval_result["avg"]
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    update=update,
                    eval_result=eval_result,
                    env_backend=env_backend,
                    observation_mode=observation_mode,
                    state_shape=state_shape,
                    score_delta_coeff=score_delta_coeff,
                )
                print(f"  >> Saved new best PPO checkpoint: " f"{best_ckpt_path}")

            if eval_result["avg"] >= target_eval_score:
                print(
                    f"\n*** TARGET REACHED! Eval avg: {eval_result['avg']:.1f} "
                    f">= {target_eval_score} ***"
                )
                break

        # ---- 6. Time budget check ------------------------------------
        elapsed = time.time() - train_start
        if time_budget_sec is not None and elapsed >= time_budget_sec:
            print(
                f"\n*** TIME BUDGET ({time_budget_sec:.0f}s) reached at update {update} "
                f"({elapsed:.1f}s elapsed) ***"
            )
            break

    # ------------------------------------------------------------------
    # Final evaluation and save
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training complete.  Running final evaluation...")
    print("=" * 60)

    @torch.no_grad()
    def final_policy_fn(s):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        logits, _ = model(s_t)
        return logits.argmax().item()

    final_eval = evaluate_policy(
        final_policy_fn,
        observation_mode=eval_observation_mode,
        n_episodes=eval_episodes,
        max_steps=eval_max_steps,
        env_backend=eval_env_backend,
        env_kwargs=eval_env_kwargs,
    )
    print(
        f"Final eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )
    save_checkpoint(
        last_ckpt_path,
        model,
        optimizer,
        update=update,
        eval_result=final_eval,
        env_backend=env_backend,
        observation_mode=observation_mode,
        state_shape=state_shape,
        score_delta_coeff=score_delta_coeff,
    )
    print(f"Saved final PPO checkpoint: {last_ckpt_path}")
    if final_eval["avg"] > best_eval_avg:
        save_checkpoint(
            best_ckpt_path,
            model,
            optimizer,
            update=update,
            eval_result=final_eval,
            env_backend=env_backend,
            observation_mode=observation_mode,
            state_shape=state_shape,
            score_delta_coeff=score_delta_coeff,
        )
        print(f"Updated best PPO checkpoint: {best_ckpt_path}")

    writer.close()
    env.close()
    save_results(algo_name, all_episode_scores, eval_result=final_eval)
    plot_training(
        all_episode_scores,
        title=f"PPO (Proximal Policy Optimization) [{env_backend}]",
        path=os.path.join(RESULTS_DIR, f"{algo_name}.png"),
        eval_scores=eval_history,
    )

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Chrome Dino")
    parser.add_argument(
        "--env-backend",
        choices=("sim", "browser"),
        default="sim",
        help="Environment backend to train on",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("feature", "image"),
        default="feature",
        help="Observation type for PPO",
    )
    parser.add_argument(
        "--eval-backend",
        choices=("sim", "browser"),
        default=None,
        help="Environment backend to evaluate on (default: same as train)",
    )
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=None,
        help="Wall-clock budget in seconds",
    )
    parser.add_argument(
        "--n-updates", type=int, default=100, help="Maximum PPO update iterations"
    )
    parser.add_argument(
        "--eval-every", type=int, default=10, help="Evaluate every N updates"
    )
    parser.add_argument(
        "--print-every", type=int, default=1, help="Print progress every N updates"
    )
    parser.add_argument(
        "--rollout-len", type=int, default=None, help="Rollout length override"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=None, help="PPO epochs override"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=None, help="Minibatch size override"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=None, help="Evaluation episodes override"
    )
    parser.add_argument(
        "--eval-max-steps", type=int, default=None, help="Evaluation max-steps override"
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize PPO weights from a checkpoint",
    )
    parser.add_argument(
        "--load-optimizer-state",
        action="store_true",
        help="Also restore optimizer state from init checkpoint",
    )
    parser.add_argument(
        "--browser-url", default="chrome://dino", help="Browser backend page URL"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show the Chrome window instead of running headless",
    )
    parser.add_argument(
        "--browser-accelerate",
        action="store_true",
        help="Use the real Dino acceleration in browser backend",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=BROWSER_IMAGE_SIZE,
        help="Image observation size for browser image mode",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=BROWSER_IMAGE_STACK,
        help="Number of grayscale frames to stack for image mode",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=BROWSER_IMAGE_ACTION_REPEAT,
        help="Repeat each browser image action for this many frames",
    )
    parser.add_argument(
        "--target-eval-score",
        type=int,
        default=TARGET_EVAL_SCORE,
        help="Stop early once deterministic eval reaches this score",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=CLIP_EPS,
        help="PPO clipping epsilon",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=ENTROPY_COEFF,
        help="Entropy bonus coefficient",
    )
    parser.add_argument(
        "--value-coeff",
        type=float,
        default=VALUE_COEFF,
        help="Value loss coefficient",
    )
    parser.add_argument(
        "--score-delta-coeff",
        type=float,
        default=SCORE_DELTA_COEFF,
        help="Reward shaping multiplier for score gains",
    )
    args = parser.parse_args()

    browser_kwargs = {
        "browser_headless": not args.show_browser,
        "browser_accelerate": args.browser_accelerate,
        "browser_url": args.browser_url,
    }
    train_kwargs = {}
    eval_kwargs = {}
    if args.env_backend == "browser":
        train_kwargs.update(browser_kwargs)
        if args.observation_mode == "image":
            train_kwargs.update(
                {
                    "image_size": args.image_size,
                    "frame_stack": args.frame_stack,
                    "action_repeat": args.action_repeat,
                }
            )
    if (args.eval_backend or args.env_backend) == "browser":
        eval_kwargs.update(browser_kwargs)
        if args.observation_mode == "image":
            eval_kwargs.update(
                {
                    "image_size": args.image_size,
                    "frame_stack": args.frame_stack,
                    "action_repeat": args.action_repeat,
                }
            )

    train(
        n_updates=args.n_updates,
        eval_every=args.eval_every,
        print_every=args.print_every,
        time_budget_sec=args.time_budget_sec,
        env_backend=args.env_backend,
        observation_mode=args.observation_mode,
        env_kwargs=train_kwargs,
        eval_env_backend=args.eval_backend,
        eval_observation_mode=args.observation_mode,
        eval_env_kwargs=eval_kwargs,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        init_checkpoint_path=args.init_checkpoint,
        load_optimizer_state=args.load_optimizer_state,
        target_eval_score=args.target_eval_score,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        score_delta_coeff=args.score_delta_coeff,
    )
