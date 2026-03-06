"""
REINFORCE with Baseline  (Phase 4)
===================================

Reference: Sutton & Barto, Chapter 13.4 — "REINFORCE with Baseline"

Overview
--------
Plain REINFORCE (Ch 13.3) uses the full return G_t to weight the policy
gradient.  The problem is that G_t can vary enormously from step to step, which
makes the gradient estimates very noisy (high variance).  Training converges
slowly because the optimizer has to average out all that noise.

The key insight of this chapter is:

    We can subtract ANY function b(s) from G_t — as long as b does not depend
    on the action a_t — and the gradient estimator stays **unbiased**.

Why unbiased?  The policy gradient theorem says:

    ∇J(θ) = E_π [ G_t · ∇ log π(a_t|s_t; θ) ]

If we subtract a baseline b(s_t):

    E_π [ (G_t - b(s_t)) · ∇ log π(a_t|s_t; θ) ]

The subtracted term expands to:

    E_π [ b(s_t) · ∇ log π(a_t|s_t; θ) ]
  = Σ_a  π(a|s_t) · b(s_t) · ∇ log π(a|s_t; θ)  / π(a|s_t) · π(a|s_t)
        (... applying the log-derivative trick in reverse ...)
  = b(s_t) · ∇ Σ_a π(a|s_t; θ)
  = b(s_t) · ∇ 1
  = 0

So the baseline term vanishes in expectation — no bias is introduced.

Choosing the baseline
---------------------
The BEST baseline (in a mean-squared-error sense) is the state-value function:

    b(s) = V^π(s) ≈ E_π[ G_t | s_t = s ]

With this choice the weighted quantity becomes the **advantage**:

    A_t = G_t - V(s_t)

Intuitive interpretation:
- A_t > 0  →  this action led to a BETTER-than-average return → reinforce it
- A_t < 0  →  this action led to a WORSE-than-average return  → discourage it
- A_t ≈ 0  →  roughly average outcome → small or no gradient push

This centered signal dramatically reduces variance compared to raw G_t, making
learning faster and more stable.

Architecture: two SEPARATE networks
------------------------------------
We maintain two independent function approximators:

    1. Policy network  π(a|s; θ)   — outputs action probabilities
    2. Value network    V(s; w)     — outputs a scalar state-value estimate

These networks do NOT share parameters.  This is the simplest correct design:
- The policy loss uses (G_t - V(s_t).detach()) · log π so that gradients only
  flow into θ.  The .detach() is critical: V is used as a **fixed** baseline
  for the policy update, not as a differentiable part of the policy objective.
- The value loss is a standard regression loss: (G_t - V(s_t))² with gradients
  flowing into w only.

Sharing a feature backbone is possible but complicates gradient flow and can
cause interference between the two objectives.  Separate nets keep things clean.

Update equations (per episode)
------------------------------
After collecting a full episode {s_0, a_0, r_1, s_1, a_1, r_2, ...} we:

  1. Compute discounted returns:  G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ...
  2. Compute advantages:          A_t = G_t - V(s_t; w)     (V detached for policy)
  3. Policy loss:    L_π = - Σ_t  A_t · log π(a_t | s_t; θ)
  4. Value loss:     L_V =   Σ_t  (G_t - V(s_t; w))²

Note that both losses loop over the same episode data but update different
parameter sets (θ vs w) via separate optimizers.  This is still a Monte-Carlo
method — we wait for the episode to finish to get exact G_t values.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Import shared infrastructure from the common module.
# DinoFeatureEnv wraps the Chrome Dino game and returns 8-dimensional feature
# vectors as state observations.  evaluate(), plot_training(), and
# save_results() are utility functions shared across all algorithm files.
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
# Device selection — use GPU when available, otherwise fall back to CPU.
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Policy Network  π(a|s; θ)
# ============================================================================
class PolicyNet(nn.Module):
    """
    Two-layer MLP that outputs a probability distribution over actions.

    Architecture:
        Input (FEATURE_DIM=8) → Linear(128) → ReLU
                               → Linear(128) → ReLU
                               → Linear(ACTION_SIZE=2) → Softmax

    The softmax ensures outputs sum to 1 and can be interpreted as
    π(a=0|s) and π(a=1|s).  We use Categorical to sample from this.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
        )

    def forward(self, x):
        """Return action log-probabilities (used by Categorical)."""
        logits = self.net(x)
        return logits

    def get_distribution(self, state):
        """
        Given a state tensor, return a Categorical distribution over actions.

        We work with logits internally and let Categorical handle the softmax
        via the `logits` argument.  This is numerically more stable than
        manually computing softmax then passing probabilities.
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

    def select_action(self, state_np):
        """
        Select an action by sampling from the policy.

        Args:
            state_np: numpy array of shape (FEATURE_DIM,)

        Returns:
            (action, log_prob) — action is an int, log_prob is a scalar tensor
        """
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        dist = self.get_distribution(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(0)


# ============================================================================
# Value Network  V(s; w)
# ============================================================================
class ValueNet(nn.Module):
    """
    Two-layer MLP that outputs a scalar estimate of the state value V(s).

    Architecture:
        Input (FEATURE_DIM=8) → Linear(128) → ReLU
                               → Linear(128) → ReLU
                               → Linear(1)

    No activation on the final layer — V(s) can be any real number.

    This network is trained to minimise MSE between its predictions and the
    actual discounted returns G_t observed from episodes.  It is ONLY used as
    a baseline for the policy gradient — its parameters w are completely
    independent from the policy parameters θ.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """Return scalar value estimate V(s)."""
        return self.net(x).squeeze(-1)


# ============================================================================
# Return computation
# ============================================================================
def compute_returns(rewards, gamma):
    """
    Compute discounted returns G_t for each timestep in an episode.

    G_t = r_{t+1} + γ · r_{t+2} + γ² · r_{t+3} + ...

    We compute this efficiently by iterating backwards:
        G_T     = 0                 (nothing after terminal)
        G_{T-1} = r_T
        G_{T-2} = r_{T-1} + γ · G_{T-1}
        ...
        G_t     = r_{t+1} + γ · G_{t+1}

    Args:
        rewards: list of floats, rewards collected during the episode
        gamma:   discount factor in [0, 1]

    Returns:
        numpy array of discounted returns, one per timestep
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


# ============================================================================
# Training loop
# ============================================================================
def train(
    n_episodes=2000,
    policy_lr=0.001,
    value_lr=0.001,
    gamma=0.99,
    print_every=10,
    eval_every=100,
    max_steps=25000,
    writer=None,
):
    """
    Train REINFORCE with Baseline on the Dino environment.

    Hyperparameters:
        policy_lr : learning rate for the policy network (Adam)
        value_lr  : learning rate for the value network  (Adam)
        gamma     : discount factor — 0.99 weights future rewards strongly
        n_episodes: total number of training episodes

    Training procedure (each episode):
        1. Roll out a full episode under the current policy.
        2. Compute discounted returns G_t for every timestep.
        3. Compute value predictions V(s_t) for every state visited.
        4. Advantage = G_t - V(s_t).detach()
              .detach() prevents gradients from the policy loss flowing
              into the value network — the value network has its own loss.
        5. Policy loss  = - Σ_t  advantage_t · log π(a_t|s_t; θ)
           Value loss   =   Σ_t  (G_t - V(s_t; w))²
        6. Backpropagate each loss through its respective optimizer.
    """
    env = DinoFeatureEnv()

    # -----------------------------------------------------------------------
    # Instantiate SEPARATE networks and optimizers.
    # Using separate Adam optimizers ensures that the policy and value
    # updates do not interfere with each other's momentum / adaptive LR.
    # -----------------------------------------------------------------------
    policy_net = PolicyNet().to(device)
    value_net = ValueNet().to(device)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    if writer is None:
        writer = create_writer('reinforce_baseline')

    # Tracking
    train_scores = []
    eval_results = []  # list of (episode, avg_score)

    for episode in range(1, n_episodes + 1):
        # -------------------------------------------------------------------
        # Step 1: Collect a full episode
        # -------------------------------------------------------------------
        state = env.reset()
        log_probs = []
        rewards = []
        states = []

        for _ in range(max_steps):
            states.append(state)

            # Sample action from the current policy
            action, log_prob = policy_net.select_action(state)
            log_probs.append(log_prob)

            state, reward, done, info = env.step(action)
            rewards.append(reward)

            if done:
                break

        episode_score = info['score']
        train_scores.append(episode_score)

        # -------------------------------------------------------------------
        # Step 2: Compute discounted returns G_t
        # -------------------------------------------------------------------
        returns = compute_returns(rewards, gamma)
        returns_t = torch.FloatTensor(returns).to(device)

        # -------------------------------------------------------------------
        # Step 3: Compute value predictions for all visited states
        # -------------------------------------------------------------------
        states_t = torch.FloatTensor(np.array(states)).to(device)
        values = value_net(states_t)  # shape: (T,)

        # -------------------------------------------------------------------
        # Step 4: Compute advantages
        #
        #   A_t = G_t - V(s_t)
        #
        # CRITICAL: .detach() on the value predictions when computing the
        # policy loss.  Without detach(), the policy loss gradient would
        # flow backwards through V(s_t) and corrupt the value network
        # weights via the policy optimizer.
        #
        # The value network should ONLY be updated via its own MSE loss.
        # -------------------------------------------------------------------
        advantages = returns_t - values.detach()

        # -------------------------------------------------------------------
        # Step 5: Compute losses
        #
        # Policy loss:
        #   L_π = - Σ_t  A_t · log π(a_t | s_t; θ)
        #
        # The negative sign is because we want to MAXIMISE expected return
        # but optimizers MINIMISE loss.  Multiplying by the advantage means:
        #   - If A_t > 0 (better than baseline): increase log π(a_t|s_t)
        #   - If A_t < 0 (worse than baseline):  decrease log π(a_t|s_t)
        #
        # Value loss:
        #   L_V = Σ_t  (G_t - V(s_t; w))²
        #
        # Standard MSE regression — push V(s_t) towards the observed G_t.
        # We use mean instead of sum for numerical stability (doesn't affect
        # the direction of the gradient, only the effective learning rate).
        # -------------------------------------------------------------------
        log_probs_t = torch.stack(log_probs)
        policy_loss = -(advantages * log_probs_t).sum()

        value_loss = ((returns_t - values) ** 2).sum()

        # -------------------------------------------------------------------
        # Step 6: Update networks
        #
        # Each network has its own optimizer.  We zero gradients, backprop,
        # and step independently to keep the two learning processes clean.
        # -------------------------------------------------------------------
        policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()

        writer.add_scalar('train/score', episode_score, episode)
        writer.add_scalar('train/policy_loss', policy_loss.item(), episode)
        writer.add_scalar('train/value_loss', value_loss.item(), episode)

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        if episode % print_every == 0:
            avg_recent = np.mean(train_scores[-print_every:])
            print(
                f"Episode {episode:5d} | "
                f"Score: {episode_score:6.0f} | "
                f"Avg({print_every}): {avg_recent:7.1f} | "
                f"Policy loss: {policy_loss.item():9.2f} | "
                f"Value loss: {value_loss.item():9.2f}"
            )
            writer.add_scalar('train/avg_score', avg_recent, episode)

        # -------------------------------------------------------------------
        # Periodic evaluation
        #
        # We evaluate the policy deterministically (argmax over π) to see
        # how well the agent actually performs without exploration noise.
        # -------------------------------------------------------------------
        if episode % eval_every == 0:
            # Build a deterministic policy function for evaluation:
            # pick the action with highest probability.
            def make_eval_fn(net):
                """Closure to capture current network for eval."""
                def policy_fn(state_np):
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
                        logits = net(state_t)
                        return logits.argmax(dim=-1).item()
                return policy_fn

            eval_fn = make_eval_fn(policy_net)
            result = evaluate(eval_fn, n_episodes=20)
            eval_results.append((episode, result['avg']))
            writer.add_scalar('eval/avg_score', result['avg'], episode)
            print(
                f"  >> Eval at episode {episode}: "
                f"avg={result['avg']:.1f}  "
                f"min={result['min']}  "
                f"max={result['max']}"
            )

    # -----------------------------------------------------------------------
    # Save results and plot
    # -----------------------------------------------------------------------
    final_eval = evaluate(
        make_eval_fn(policy_net), n_episodes=20
    ) if eval_results else None

    writer.close()
    save_results('reinforce_baseline', train_scores, final_eval)

    plot_path = os.path.join(
        RESULTS_DIR, 'reinforce_baseline.png'
    )
    plot_training(
        train_scores,
        title='REINFORCE with Baseline — Training Curve',
        path=plot_path,
        eval_scores=eval_results,
    )

    return policy_net, value_net, train_scores


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    print(f"Device: {device}")
    print("=" * 60)
    print("REINFORCE with Baseline  (Sutton & Barto, Ch 13.4)")
    print("=" * 60)
    print()
    print("Key idea: subtracting a learned baseline V(s) from the return G_t")
    print("reduces gradient variance WITHOUT introducing bias.")
    print()
    print("  Policy loss:  -Σ_t (G_t - V(s_t).detach()) · log π(a_t|s_t)")
    print("  Value loss:    Σ_t (G_t - V(s_t))²")
    print()

    policy_net, value_net, scores = train(
        n_episodes=2000,
        policy_lr=0.001,
        value_lr=0.001,
        gamma=0.99,
        print_every=10,
        eval_every=100,
    )

    print()
    print("Training complete.")
