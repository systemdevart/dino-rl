"""
True Online TD(λ) for the Chrome Dino Game — Phase 7
=====================================================

Sutton & Barto reference: Chapter 12.5, "True Online TD(λ)"

This file implements True Online TD(λ) with a hybrid architecture:
a neural-network feature extractor feeding a linear value head whose
weights are updated with Dutch eligibility traces.

Background: Eligibility Traces
------------------------------
Eligibility traces are one of the most fundamental mechanisms in RL.
They bridge the gap between the two extremes of temporal-difference
learning:

    TD(0):  Updates from the *immediate* next reward + bootstrapped value.
            Low variance but high bias (relies on the current value estimate).
            Fast learning per step, but only propagates information one step
            at a time.

    Monte Carlo (MC):  Updates from the *full return* — the actual sum of
            rewards to the end of the episode.  Zero bias (uses the true
            return) but high variance (each return is a single noisy sample).
            Information from the end of the episode must wait until the
            episode terminates.

Eligibility traces provide a smooth interpolation controlled by the
parameter λ (lambda), where 0 ≤ λ ≤ 1:

    λ = 0  →  Recovers TD(0).  Only the most recently visited state is
              updated, using the one-step TD error.

    λ = 1  →  Recovers Monte Carlo.  Every state visited during the episode
              is updated according to how recently it was visited, using the
              full return (in the episodic, undiscounted case).

    0 < λ < 1  →  A blend.  Recent states are updated more strongly, and
              the effective backup depth scales as 1/(1-λ).  A typical
              choice like λ = 0.9 means the trace has a half-life of about
              ~7 steps (since 0.9^7 ≈ 0.48), so the agent looks back ~7
              steps when assigning credit — far more than TD(0) but far
              less noisy than full MC.

Mechanically, an eligibility trace is a vector z_t of the same
dimensionality as the weight vector w.  At each step, z is decayed by
γλ and then incremented by the gradient of the current value with
respect to w.  When a TD error δ occurs, the weight update is
proportional to δ·z, so every recently-visited state gets credit in
proportion to how recently (and how strongly) its features were active.

Types of Eligibility Traces
----------------------------
1.  Accumulating traces:  z ← γλz + ∇V(s).  The simplest form.
    If a state is visited multiple times, its trace accumulates and can
    grow without bound.  Works well in tabular settings.

2.  Replacing traces:  z_i ← 1 for active features, z_i ← γλ z_i
    otherwise.  Caps the trace at 1 for binary features.  Useful in
    tabular / tile-coding settings but does not generalize to arbitrary
    function approximation.

3.  Dutch traces (used here):
        z ← γλz + (1 - α γλ z^T x) x
    This is the trace used in True Online TD(λ).  It includes a
    correction term (1 - α γλ z^T x) that ensures the algorithm
    produces *exactly* the same sequence of weight vectors as the
    theoretically ideal offline λ-return algorithm, but computed
    fully online (incrementally, step by step).

True Online TD(λ)
------------------
The "offline λ-return" algorithm is conceptually clean: at the end of
an episode, for each time step t, compute the λ-return:

    G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t:t+n  +  λ^{T-t-1} G_t

and update the weights toward each G_t^λ.  This is the "ideal" way to
use eligibility traces, but it requires the entire episode before any
update can occur.

True Online TD(λ) (van Seijen et al., 2016) is the remarkable result
that this offline computation can be replicated *exactly* by an online,
incremental algorithm.  The key insight is maintaining an auxiliary
scalar V_old (the value of the current state *before* the weight
update) and using the Dutch trace.  The full update at each step is:

    x       = features(s)
    x'      = features(s')
    V_s     = w^T x          (current value of s)
    δ       = r + γ w^T x' - V_s
    z       ← γλz + (1 - α γλ z^T x) x      (Dutch trace)
    w       ← w + α(δ + V_s - V_old) z - α(V_s - V_old) x
    V_old   ← w^T x'        (for next step)

This is a *provably* exact online equivalent of the offline λ-return
algorithm for linear function approximation.

Extension to Function Approximation (Hybrid Architecture)
-----------------------------------------------------------
True Online TD(λ) with Dutch traces is derived for *linear* function
approximation: V(s) = w^T φ(s).  Extending it to deep nonlinear
networks is non-trivial because:

    - The trace z must have the same shape as w, so for a large NN the
      trace vector would be enormous.
    - The theoretical guarantees (exact equivalence to offline λ-return)
      hold only for the linear case.

A practical compromise (used here) is to split the architecture:

    1.  A nonlinear neural-network feature extractor  φ_θ(s) : R^8 → R^128
        This is a 2-layer ReLU network that learns a rich representation.
        Its parameters θ are updated with *standard backprop* using a
        semi-gradient TD loss, periodically (e.g., once per episode).

    2.  A linear value head  V(s) = w^T φ_θ(s)
        This is a single linear layer (no bias, or with bias).  The
        True Online TD(λ) algorithm — including the Dutch trace vector z —
        operates on w only.  Since w is a small vector (128 + 1 dims),
        the trace is cheap to maintain and the theoretical guarantees
        of exact online λ-return equivalence apply (conditional on
        fixed features).

This is sometimes called a "semi-linear" architecture and is a common
pragmatic approach for combining deep representation learning with
eligibility-trace-based algorithms.

Policy: Epsilon-Greedy with Value-Based Action Preference
----------------------------------------------------------
Since we are learning V(s) rather than Q(s,a), we cannot directly pick
the action with the highest value.  We use a simple heuristic: maintain
running exponential averages of V(s') for each action, and pick the
action whose average V(s') is higher.  Under epsilon-greedy, we choose
randomly with probability ε and greedily otherwise.
"""

import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Import shared infrastructure
# ---------------------------------------------------------------------------
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
# Device detection
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TD(λ)] Using device: {device}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
GAMMA = 0.99          # Discount factor
LAMBDA = 0.9          # Trace-decay parameter (λ). 0→TD(0), 1→MC
LR_LINEAR = 0.001     # Step-size α for the linear value head (True Online TD(λ))
LR_FEATURES = 0.001   # Learning rate for the NN feature extractor (Adam)
EPSILON_START = 1.0    # Initial exploration rate
EPSILON_END = 0.01     # Minimum exploration rate
EPSILON_DECAY = 0.998  # Multiplicative decay per episode
HIDDEN_DIM = 128       # Width of each hidden layer in the feature extractor
EMA_BETA = 0.01        # Exponential moving average rate for action-value tracking
FEATURE_UPDATE_FREQ = 1  # Update NN feature extractor every N episodes


# =========================================================================
# Neural Network Feature Extractor
# =========================================================================
class FeatureExtractor(nn.Module):
    """
    2-layer ReLU network that maps raw 8-dim environment features into
    a learned 128-dim representation.

    This is the nonlinear part of our hybrid architecture.  Its weights θ
    are updated periodically via standard backprop, *not* via eligibility
    traces (traces are reserved for the linear value head).
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (input_dim,) raw features.
        Returns:
            (batch, hidden_dim) or (hidden_dim,) learned features.
        """
        return self.net(x)


# =========================================================================
# Linear Value Head (with True Online TD(λ) updates)
# =========================================================================
class LinearValueHead:
    """
    V(s) = w^T φ(s) + b

    The weight vector w (and bias b) live as plain numpy arrays so that
    we can apply the True Online TD(λ) update rule directly.  We maintain
    the Dutch eligibility trace z of the same shape as the full parameter
    vector [w; b].

    Why numpy instead of torch?
    ---------------------------
    The True Online TD(λ) update is a hand-coded formula (not a loss
    function), so there is no advantage to automatic differentiation.
    Keeping w in numpy avoids the overhead of torch graph construction
    and makes the update equations clearer.
    """

    def __init__(self, feature_dim: int = HIDDEN_DIM):
        self.dim = feature_dim

        # Weight vector (feature_dim,) and scalar bias
        self.w = np.zeros(feature_dim, dtype=np.float64)
        self.b = 0.0

        # Dutch eligibility trace — same shape as [w, b]
        self.z_w = np.zeros(feature_dim, dtype=np.float64)
        self.z_b = 0.0

        # V_old: the value of the *current* state computed *before* the
        # weight update at the previous step.  Initialized to 0 at the
        # start of each episode.
        self.v_old = 0.0

    def value(self, phi: np.ndarray) -> float:
        """Compute V(s) = w^T φ(s) + b."""
        return float(np.dot(self.w, phi) + self.b)

    def reset_traces(self):
        """
        Reset traces and V_old at the start of each episode.

        Eligibility traces should *not* carry over between episodes because
        they encode temporal credit within a single trajectory.  At episode
        boundaries, we start fresh.
        """
        self.z_w[:] = 0.0
        self.z_b = 0.0
        self.v_old = 0.0

    def update(self, phi: np.ndarray, phi_next: np.ndarray,
               reward: float, done: bool, alpha: float,
               gamma: float = GAMMA, lam: float = LAMBDA):
        """
        True Online TD(λ) update (Sutton & Barto, Ch 12.5, Box on p. 307).

        Args:
            phi:      Feature vector of current state s, shape (dim,).
            phi_next: Feature vector of next state s', shape (dim,).
            reward:   Reward received on the transition s → s'.
            done:     Whether s' is terminal.
            alpha:    Step-size parameter.
            gamma:    Discount factor.
            lam:      Trace-decay parameter λ.

        The update equations (for the weight vector w; bias analogous):

            v_s     = w^T φ
            v_sp    = w^T φ'  (0 if done)
            δ       = r + γ v_sp - v_s

            # Dutch trace (key to True Online TD(λ)):
            z       ← γ λ z + (1 - α γ λ z^T φ) φ
            #
            # The correction term (1 - α γ λ z^T φ) distinguishes the
            # Dutch trace from a standard accumulating trace.  It prevents
            # double-counting and ensures exact equivalence to the offline
            # λ-return algorithm.

            # Weight update:
            w       ← w + α (δ + v_s - v_old) z - α (v_s - v_old) φ
            #
            # The (v_s - v_old) terms are the True Online correction.
            # v_old is the value of s computed with the weights from
            # *before* the previous update.  This correction accounts for
            # the fact that updating w changes V(s) even though we already
            # used V(s) in the TD error.

            v_old   ← v_sp   (saved for the next step)
        """
        # Current state value (with current weights, before this update)
        v_s = self.value(phi)

        # Next state value (0 if terminal)
        v_sp = 0.0 if done else self.value(phi_next)

        # TD error
        delta = reward + gamma * v_sp - v_s

        # ---------------------------------------------------------------
        # Dutch trace update for weights
        # z ← γλz + (1 - α γλ z^T φ) φ
        # ---------------------------------------------------------------
        z_dot_phi = np.dot(self.z_w, phi)
        coeff = 1.0 - alpha * gamma * lam * z_dot_phi
        self.z_w = gamma * lam * self.z_w + coeff * phi

        # Dutch trace update for bias (φ_bias = 1 always)
        z_b_dot_1 = self.z_b  # z^T [1] for the bias dimension
        coeff_b = 1.0 - alpha * gamma * lam * z_b_dot_1
        self.z_b = gamma * lam * self.z_b + coeff_b * 1.0

        # ---------------------------------------------------------------
        # Weight update
        # w ← w + α(δ + v_s - v_old) z - α(v_s - v_old) φ
        # ---------------------------------------------------------------
        correction = v_s - self.v_old
        self.w += alpha * (delta + correction) * self.z_w - alpha * correction * phi
        self.b += alpha * (delta + correction) * self.z_b - alpha * correction * 1.0

        # Save v_old for next step
        self.v_old = v_sp

        return delta


# =========================================================================
# TD(λ) Agent
# =========================================================================
class TDLambdaAgent:
    """
    Agent combining:
      - NN feature extractor (trained with standard backprop TD loss)
      - Linear value head (trained with True Online TD(λ) / Dutch traces)
      - Epsilon-greedy action selection with value-based preference
    """

    def __init__(self):
        # Neural network feature extractor
        self.feature_net = FeatureExtractor().to(device)
        self.feature_optimizer = optim.Adam(
            self.feature_net.parameters(), lr=LR_FEATURES
        )

        # Linear value head (numpy-based for clean trace updates)
        self.value_head = LinearValueHead(feature_dim=HIDDEN_DIM)

        # Exploration
        self.epsilon = EPSILON_START

        # Running EMA of V(s') for each action, used for action selection.
        # action_value_ema[a] ≈ E[ V(s') | we took action a ]
        # By picking the action whose next-state value is higher, we
        # implement a simple 1-step lookahead without needing a model.
        self.action_value_ema = np.zeros(ACTION_SIZE, dtype=np.float64)

        # Experience buffer for feature extractor training.
        # We collect (state, reward, next_state, done) tuples during
        # the episode and train the NN at the end.
        self.episode_buffer = []

    def _extract_features_np(self, state: np.ndarray) -> np.ndarray:
        """
        Pass a raw state through the NN feature extractor and return
        a numpy vector.  No gradient tracking.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            phi = self.feature_net(state_t).squeeze(0).cpu().numpy()
        return phi.astype(np.float64)

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection based on action-value EMAs.

        With probability ε, choose uniformly at random.
        Otherwise, choose the action whose running average V(s') is higher.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(ACTION_SIZE)

        # Greedy: pick action with higher EMA of successor values
        return int(np.argmax(self.action_value_ema))

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Deterministic greedy action for evaluation."""
        return int(np.argmax(self.action_value_ema))

    def update_action_ema(self, action: int, v_next: float):
        """
        Update the exponential moving average of V(s') for the given action.

        This tracks:  action_value_ema[a] ≈ E[ V(s') | action = a ]

        By comparing these EMAs, the agent learns which action tends to
        lead to higher-valued successor states, effectively performing a
        1-step lookahead on V(s).
        """
        self.action_value_ema[action] = (
            (1 - EMA_BETA) * self.action_value_ema[action]
            + EMA_BETA * v_next
        )

    def decay_epsilon(self):
        """Multiplicative epsilon decay, clamped at EPSILON_END."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def train_feature_extractor(self):
        """
        Train the NN feature extractor using semi-gradient TD(0) loss
        on the buffered episode transitions.

        Loss = Σ_t (V_predicted(s_t) - [r_t + γ V_target(s_{t+1})])^2

        where V_predicted uses the NN + linear head, and the target is
        computed with the *current* linear head weights but detached
        features for the next state (semi-gradient: we don't differentiate
        through the target).

        After this update, the feature representations change, so the
        linear head's weights (and traces) are operating on a slightly
        different feature space.  This is why we reset traces each episode
        and only update features periodically.
        """
        if len(self.episode_buffer) < 2:
            return

        # Prepare batch tensors
        states = np.array([t[0] for t in self.episode_buffer])
        rewards = np.array([t[1] for t in self.episode_buffer], dtype=np.float32)
        next_states = np.array([t[2] for t in self.episode_buffer])
        dones = np.array([t[3] for t in self.episode_buffer], dtype=np.float32)

        states_t = torch.FloatTensor(states).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        # Linear head weights as torch tensors (frozen; we only train the NN)
        w_t = torch.DoubleTensor(self.value_head.w).to(device).float()
        b_val = float(self.value_head.b)

        # Forward pass: compute learned features
        phi = self.feature_net(states_t)          # (T, hidden_dim)
        v_pred = (phi * w_t.unsqueeze(0)).sum(dim=1) + b_val   # (T,)

        # Target: r + γ V(s') with detached next-state features
        with torch.no_grad():
            phi_next = self.feature_net(next_states_t)
            v_next = (phi_next * w_t.unsqueeze(0)).sum(dim=1) + b_val
            target = rewards_t + GAMMA * v_next * (1.0 - dones_t)

        loss = nn.functional.mse_loss(v_pred, target)

        self.feature_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.feature_net.parameters(), max_norm=1.0)
        self.feature_optimizer.step()

        return loss.item()


# =========================================================================
# Training Loop
# =========================================================================
def train(n_episodes: int = 2000, print_every: int = 10, eval_every: int = 100, writer=None):
    """
    Train the True Online TD(λ) agent on the Chrome Dino game.

    The training loop proceeds as follows each episode:
      1. Reset the environment and eligibility traces.
      2. For each step in the episode:
         a. Extract NN features φ(s) from the raw state.
         b. Select action (ε-greedy based on action-value EMAs).
         c. Take a step, observe reward, next state.
         d. Extract NN features φ(s') from the next state.
         e. Apply True Online TD(λ) update to the linear value head.
         f. Update action-value EMA for the chosen action.
         g. Store transition for later NN training.
      3. At the end of the episode:
         - Optionally train the NN feature extractor on buffered data.
         - Decay epsilon.
    """
    env = DinoFeatureEnv()
    agent = TDLambdaAgent()

    if writer is None:
        writer = create_writer('td_lambda')

    all_scores = []
    eval_history = []  # (episode, avg_eval_score)
    recent_scores = deque(maxlen=100)

    best_eval_score = 0

    print(f"[TD(λ)] Starting training: {n_episodes} episodes")
    print(f"[TD(λ)] γ={GAMMA}, λ={LAMBDA}, α_linear={LR_LINEAR}, "
          f"α_features={LR_FEATURES}, ε={EPSILON_START}→{EPSILON_END}")
    print(f"[TD(λ)] Feature extractor: {FEATURE_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM} (ReLU)")
    print(f"[TD(λ)] Linear value head: {HIDDEN_DIM}→1")
    total_params = sum(p.numel() for p in agent.feature_net.parameters())
    print(f"[TD(λ)] Feature extractor parameters: {total_params:,}")
    print(f"[TD(λ)] Linear head parameters: {HIDDEN_DIM + 1} (w) + {HIDDEN_DIM + 1} (z)")
    print()

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.value_head.reset_traces()
        agent.episode_buffer.clear()

        # Extract initial features
        phi = agent._extract_features_np(state)

        episode_reward = 0.0
        steps = 0

        for t in range(25000):
            # Select action
            action = agent.select_action(state)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Extract next-state features
            phi_next = agent._extract_features_np(next_state)

            # -----------------------------------------------------------------
            # True Online TD(λ) update on the linear value head
            # -----------------------------------------------------------------
            # This is the core algorithm.  See the LinearValueHead.update()
            # docstring for the full equations.
            agent.value_head.update(
                phi=phi,
                phi_next=phi_next,
                reward=reward,
                done=done,
                alpha=LR_LINEAR,
            )

            # Update action-value EMA for action selection
            v_next = 0.0 if done else agent.value_head.value(phi_next)
            agent.update_action_ema(action, v_next)

            # Store transition for feature extractor training
            agent.episode_buffer.append((state, reward, next_state, done))

            episode_reward += reward
            steps += 1
            state = next_state
            phi = phi_next

            if done:
                break

        score = info['score']
        all_scores.append(score)
        recent_scores.append(score)
        writer.add_scalar('train/score', score, episode)

        # Train the NN feature extractor periodically
        if episode % FEATURE_UPDATE_FREQ == 0:
            agent.train_feature_extractor()

        # Decay exploration
        agent.decay_epsilon()

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------
        if episode % print_every == 0:
            avg = np.mean(recent_scores)
            print(
                f"Ep {episode:4d}/{n_episodes} | "
                f"Score: {score:5d} | "
                f"Avg100: {avg:7.1f} | "
                f"Steps: {steps:5d} | "
                f"ε: {agent.epsilon:.4f} | "
                f"V_ema: [{agent.action_value_ema[0]:.2f}, "
                f"{agent.action_value_ema[1]:.2f}]"
            )
            writer.add_scalar('train/avg_score', avg, episode)
            writer.add_scalar('train/epsilon', agent.epsilon, episode)

        # -----------------------------------------------------------------
        # Periodic evaluation
        # -----------------------------------------------------------------
        if episode % eval_every == 0:
            # Build a greedy policy function for evaluation
            # We need a closure that captures the current agent state.
            # During eval, we use the same EMA-based greedy selection.
            def eval_policy(s, _agent=agent):
                return _agent.select_action_greedy(s)

            eval_result = evaluate(eval_policy, n_episodes=20)
            eval_avg = eval_result['avg']
            eval_history.append((episode, eval_avg))
            writer.add_scalar('eval/avg_score', eval_avg, episode)

            if eval_avg > best_eval_score:
                best_eval_score = eval_avg

            print(
                f"  >> EVAL (20 ep): "
                f"Avg {eval_avg:.1f} | "
                f"Min {eval_result['min']} | "
                f"Max {eval_result['max']} | "
                f"Best {best_eval_score:.1f}"
            )

    # -----------------------------------------------------------------
    # Save results and plot
    # -----------------------------------------------------------------
    print(f"\n[TD(λ)] Training complete.")
    print(f"[TD(λ)] Final avg (last 100): {np.mean(all_scores[-100:]):.1f}")
    print(f"[TD(λ)] Best eval score: {best_eval_score:.1f}")

    writer.close()
    save_results('td_lambda', all_scores, eval_result)
    plot_training(
        all_scores,
        title='True Online TD(λ) — Chrome Dino',
        path=os.path.join(RESULTS_DIR, 'td_lambda.png'),
        eval_scores=eval_history,
    )


# =========================================================================
# Entry Point
# =========================================================================
if __name__ == '__main__':
    train(n_episodes=2000, print_every=10, eval_every=100)
