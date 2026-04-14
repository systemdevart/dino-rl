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
ROLLOUT_LEN = 40000   # Must be > 32,600 frames needed for score 10k
MINIBATCH_SIZE = 512
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
TARGET_EVAL_SCORE = 10000


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic network for PPO.

    Architecture:
        state (FEATURE_DIM=8)
            -> Linear(128) -> ReLU
            -> Linear(128) -> ReLU          <-- shared backbone
            +--> Linear(ACTION_SIZE=2)      <-- policy head (actor)
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

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Shared backbone -- three hidden layers with ReLU activation.
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Policy head (actor): outputs raw logits for each action.
        # We do NOT apply softmax here; torch.distributions.Categorical
        # handles log-softmax internally for numerical stability.
        self.policy_head = nn.Linear(128, action_dim)

        # Value head (critic): outputs a single scalar V(s).
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x: state tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            logits: action logits of shape (batch, action_dim) or (action_dim,)
            value:  state value of shape (batch, 1) or (1,)
        """
        features = self.backbone(x)
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

    def __init__(self, rollout_len: int, state_dim: int, device: str):
        self.rollout_len = rollout_len
        self.device = device
        self.ptr = 0  # current write position

        # Pre-allocate tensors for efficiency.
        self.states = torch.zeros(rollout_len, state_dim, device=device)
        self.actions = torch.zeros(rollout_len, dtype=torch.long, device=device)
        self.rewards = torch.zeros(rollout_len, device=device)
        self.dones = torch.zeros(rollout_len, device=device)
        self.log_probs = torch.zeros(rollout_len, device=device)
        self.values = torch.zeros(rollout_len, device=device)

        # These are computed after the rollout is complete.
        self.advantages = torch.zeros(rollout_len, device=device)
        self.returns = torch.zeros(rollout_len, device=device)

    def store(self, state, action, reward, done, log_prob, value):
        """Store one transition at the current pointer position."""
        idx = self.ptr
        self.states[idx] = state
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
    model: ActorCritic,
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
            loss = (
                policy_loss
                + value_coeff * value_loss
                + entropy_coeff * entropy_loss
            )

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
                approx_kl = (0.5 * (log_ratio ** 2)).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_loss.item()  # flip sign for readability
            total_clip_frac += clipped
            total_approx_kl += approx_kl
            total_ratio_mean += ratio.mean().item()
            n_updates += 1

    n = max(n_updates, 1)
    return {
        'policy_loss': total_policy_loss / n,
        'value_loss': total_value_loss / n,
        'entropy': total_entropy / n,
        'clip_frac': total_clip_frac / n,
        'approx_kl': total_approx_kl / n,
        'ratio_mean': total_ratio_mean / n,
        'adv_mean': raw_adv_mean,
        'adv_std': raw_adv_std,
        'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
    }


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout(
    env: DinoFeatureEnv,
    model: ActorCritic,
    buffer: RolloutBuffer,
    state: np.ndarray,
    device: str,
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

    for _step in range(buffer.rollout_len):
        state_t = torch.tensor(state, dtype=torch.float32, device=device)

        # Get action, log_prob, and value from the current policy.
        action, log_prob, value = model.get_action_and_value(state_t)

        # Take the action in the environment.
        next_state, reward, done, info = env.step(action.item())

        # Store the transition.
        buffer.store(
            state=state_t,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        if done:
            # Episode ended.  Record the score and reset.
            episode_scores.append(info['score'])
            state = env.reset()
        else:
            state = next_state

    # Compute V(s_T) for bootstrapping the final advantage.
    # If the last step was terminal, the GAE computation will zero this out
    # via the (1 - done) factor, but we still need a value for the math.
    last_state_t = torch.tensor(state, dtype=torch.float32, device=device)
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"PPO  |  device={device}  lr={LR}  clip_eps={CLIP_EPS}")
    print(f"       gamma={GAMMA}  lam_gae={LAM_GAE}  epochs={PPO_EPOCHS}")
    print(f"       rollout_len={ROLLOUT_LEN}  minibatch={MINIBATCH_SIZE}")
    print(f"       entropy_coeff={ENTROPY_COEFF}  value_coeff={VALUE_COEFF}")
    print(f"       total steps = {n_updates} * {ROLLOUT_LEN} = {n_updates * ROLLOUT_LEN}")
    print()

    env = DinoFeatureEnv()
    model = ActorCritic(FEATURE_DIM, ACTION_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # Linear LR annealing: decay from LR to 0 over n_updates
    def lr_lambda(update):
        return 1.0 - update / n_updates
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    buffer = RolloutBuffer(ROLLOUT_LEN, FEATURE_DIM, device)

    if writer is None:
        writer = create_writer('ppo')

    # Tracking
    all_episode_scores: list[int] = []
    eval_history: list[tuple[int, float]] = []  # (update_num, avg_score)
    total_steps = 0

    # Initial state
    state = env.reset()
    train_start = time.time()

    # ------------------------------------------------------------------
    # PPO update loop
    # ------------------------------------------------------------------
    for update in range(1, n_updates + 1):
        # ---- 1. Collect rollout --------------------------------------
        model.eval()
        state, episode_scores = collect_rollout(
            env, model, buffer, state, device
        )
        model.train()

        total_steps += ROLLOUT_LEN
        all_episode_scores.extend(episode_scores)

        # ---- 2. Anneal learning rate ---------------------------------
        scheduler.step()

        # ---- 3. PPO parameter update ---------------------------------
        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            clip_eps=CLIP_EPS,
            ppo_epochs=PPO_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            entropy_coeff=ENTROPY_COEFF,
            value_coeff=VALUE_COEFF,
        )

        # ---- 3. Logging ----------------------------------------------
        if update % print_every == 0:
            # Average score from episodes completed during this rollout.
            if episode_scores:
                avg_score = np.mean(episode_scores)
                n_eps = len(episode_scores)
            else:
                avg_score = 0.0
                n_eps = 0

            cur_lr = optimizer.param_groups[0]['lr']
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
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], update)
            writer.add_scalar('train/value_loss', metrics['value_loss'], update)
            writer.add_scalar('train/entropy', metrics['entropy'], update)
            writer.add_scalar('train/avg_score', avg_score, update)
            writer.add_scalar('train/clip_fraction', metrics['clip_frac'], update)
            writer.add_scalar('train/approx_kl', metrics['approx_kl'], update)
            writer.add_scalar('train/ratio_mean', metrics['ratio_mean'], update)
            writer.add_scalar('train/advantage_mean', metrics['adv_mean'], update)
            writer.add_scalar('train/advantage_std', metrics['adv_std'], update)
            writer.add_scalar('train/grad_norm', metrics['grad_norm'], update)
            writer.add_scalar('train/learning_rate', cur_lr, update)
            if episode_scores:
                writer.add_scalar('train/max_score', max(episode_scores), update)
                writer.add_scalar('train/min_score', min(episode_scores), update)

        # ---- 4. Periodic evaluation ----------------------------------
        if update % eval_every == 0:
            @torch.no_grad()
            def policy_fn(s, _model=model, _device=device):
                s_t = torch.tensor(s, dtype=torch.float32, device=_device)
                logits, _ = _model(s_t)
                return logits.argmax().item()

            eval_result = evaluate(policy_fn)
            eval_history.append((len(all_episode_scores), eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], update)
            print(
                f"  >> Eval @ update {update}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

            if eval_result['avg'] >= TARGET_EVAL_SCORE:
                print(f"\n*** TARGET REACHED! Eval avg: {eval_result['avg']:.1f} >= {TARGET_EVAL_SCORE} ***")
                break

        # ---- 5. Time budget check ------------------------------------
        elapsed = time.time() - train_start
        if time_budget_sec is not None and elapsed >= time_budget_sec:
            print(f"\n*** TIME BUDGET ({time_budget_sec:.0f}s) reached at update {update} "
                  f"({elapsed:.1f}s elapsed) ***")
            break

    # ------------------------------------------------------------------
    # Final evaluation and save
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training complete.  Running final evaluation...")
    print("=" * 60)

    @torch.no_grad()
    def final_policy_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        logits, _ = model(s_t)
        return logits.argmax().item()

    final_eval = evaluate(final_policy_fn)
    print(
        f"Final eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )

    writer.close()
    save_results('ppo', all_episode_scores, eval_result=final_eval)
    plot_training(
        all_episode_scores,
        title='PPO (Proximal Policy Optimization)',
        path=os.path.join(RESULTS_DIR, 'ppo.png'),
        eval_scores=eval_history,
    )

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
