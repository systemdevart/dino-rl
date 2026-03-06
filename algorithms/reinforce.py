"""
REINFORCE -- Monte Carlo Policy Gradient
=========================================
Sutton & Barto, Ch 13.3

This file implements the simplest policy-gradient algorithm.  Unlike value-
based methods (e.g. DQN) which learn an *action-value* function Q(s,a) and
derive the policy from it (pick the action with the highest Q), policy-
gradient methods directly parameterise the policy itself:

    pi_theta(a | s)    -- a probability distribution over actions given state

and improve it by gradient *ascent* on the expected return J(theta).

Why bother?
-----------
1.  Policy-gradient methods can naturally handle stochastic policies and
    continuous action spaces.
2.  Small changes in theta produce small changes in the policy (and therefore
    in the state-visitation distribution), which tends to give smoother
    learning dynamics than the discontinuous jumps of greedy value updates.
3.  They directly optimise the thing we care about -- the expected total
    reward -- rather than a proxy (the value function).

The Policy Gradient Theorem  (Sutton & Barto, Theorem 13.1)
------------------------------------------------------------
The gradient of the objective J(theta) with respect to the policy parameters
theta is proportional to:

    nabla J(theta)  propto  sum_t  G_t  nabla ln pi_theta(a_t | s_t)

where
    G_t = sum_{k=0}^{T-t-1}  gamma^k  r_{t+k}

is the *discounted return* from timestep t onward.

Intuitively: we push up the log-probability of actions that led to *high*
returns and push down the log-probability of those that led to *low* returns,
weighted by exactly how good (G_t) each outcome was.

REINFORCE is the Monte Carlo instantiation of this idea -- we collect a full
episode, compute the *actual* returns G_t for every timestep, and use them
directly as the gradient weight.  No bootstrapping, no value function.

Variance issue
--------------
Because G_t is a *single sample* of the return (one full episode), the
gradient estimate has high variance.  The agent may need many episodes to
converge, since lucky and unlucky episodes create noisy gradient signals.
Later algorithms (actor-critic, A2C, PPO) reduce this variance by
introducing a *baseline* -- typically a learned value function V(s) -- so
the weight becomes the *advantage*  A_t = G_t - V(s_t).  REINFORCE with
a baseline is covered in Sutton & Barto Ch 13.4.

Softmax policy (discrete actions)
---------------------------------
For a discrete action space (here: {0=do-nothing, 1=jump}), the standard
parameterisation is the *softmax* policy:

    pi_theta(a | s) = exp(z_a) / sum_a' exp(z_a')

where z = f_theta(s) are the logits produced by a neural network.  The
softmax guarantees a proper probability distribution (non-negative, sums
to 1), and the gradient of log pi naturally encourages exploration because
it never assigns *zero* probability to any action.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
# Policy Network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    A simple 2-layer MLP that outputs a probability distribution over actions.

    Architecture:
        state (8 features)
            -> Linear(128) -> ReLU
            -> Linear(128) -> ReLU
            -> Linear(ACTION_SIZE)        <-- raw logits
            -> Softmax (applied externally via Categorical)

    We do NOT apply softmax inside forward(); instead we return raw logits
    and let torch.distributions.Categorical handle the numerically-stable
    log-softmax internally.  This avoids the classic log(softmax(x)) numerical
    instability and is standard PyTorch practice.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits given a state tensor."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    Compute discounted returns G_t for every timestep in an episode.

    For each timestep t in [0, T-1]:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t-1} * r_{T-1}

    We compute this efficiently by iterating *backwards* through the episode:

        G_{T-1} = r_{T-1}
        G_t     = r_t  +  gamma * G_{t+1}

    This runs in O(T) time and O(T) space.

    Args:
        rewards: list of rewards [r_0, r_1, ..., r_{T-1}]
        gamma: discount factor

    Returns:
        list of returns [G_0, G_1, ..., G_{T-1}]
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()  # now returns[t] = G_t
    return returns


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_episodes: int = 2000,
    gamma: float = 0.99,
    lr: float = 0.001,
    print_every: int = 10,
    eval_every: int = 100,
    device: str | None = None,
    writer=None,
):
    """
    Train a policy using REINFORCE on the Dino game.

    The training loop for one episode:
        1. Reset the environment.
        2. Roll out the full episode, storing (log_prob, reward) at each step.
        3. Compute returns G_t for every timestep (backward pass over rewards).
        4. Compute the REINFORCE loss:
               L = - sum_t  G_t * log pi(a_t | s_t)
           The negative sign is because optimisers *minimise* loss, but we
           want to *maximise* expected return.
        5. Backprop and update theta.

    Why the loss works:
        The gradient of L w.r.t. theta is:
            nabla L = - sum_t G_t * nabla log pi(a_t | s_t)
        Taking a gradient *descent* step on L is equivalent to a gradient
        *ascent* step on J(theta), which is exactly what the policy gradient
        theorem prescribes.

    Args:
        n_episodes: number of training episodes
        gamma: discount factor
        lr: learning rate for Adam optimiser
        print_every: print a progress line every N episodes
        eval_every: run deterministic evaluation every N episodes
        device: 'cuda' or 'cpu'; auto-detected if None
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"REINFORCE  |  device={device}  lr={lr}  gamma={gamma}")

    env = DinoFeatureEnv()
    policy = PolicyNetwork(FEATURE_DIM, ACTION_SIZE).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    if writer is None:
        writer = create_writer('reinforce')

    train_scores: list[int] = []
    eval_history: list[tuple[int, float]] = []  # (episode, avg_score)

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False

        # ---- 1. Collect a full episode --------------------------------
        while not done:
            # Convert state to tensor
            state_t = torch.tensor(state, dtype=torch.float32, device=device)

            # Forward pass: get action logits, build distribution
            logits = policy(state_t)
            dist = Categorical(logits=logits)

            # Sample action from the stochastic policy
            action = dist.sample()

            # Store log pi(a_t | s_t) for the gradient computation later.
            # .log_prob() computes log softmax(z)[a], which is numerically
            # stable because Categorical uses log_softmax internally.
            log_probs.append(dist.log_prob(action))

            # Take the action
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = next_state

        # ---- 2. Compute returns G_t -----------------------------------
        returns = compute_returns(rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Note: we do NOT normalise returns here.  While normalisation is a
        # common trick for environments with diverse per-step rewards, it is
        # harmful in sparse-reward environments like the Dino game, where all
        # steps except the terminal one share the same +0.1 reward.  Normalising
        # within a single episode washes out the difference between "good"
        # episodes (survived longer) and "bad" ones (died early), destroying
        # the gradient signal that REINFORCE needs to learn jump timing.

        # ---- 3. Compute REINFORCE loss --------------------------------
        # L = - sum_t G_t * log pi(a_t | s_t)
        #
        # We stack the per-step log probs into a single tensor so PyTorch
        # can backprop through the whole computation graph in one go.
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns_t).sum()

        # ---- 4. Gradient descent step ---------------------------------
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Bookkeeping ---------------------------------------------
        score = info['score']
        train_scores.append(score)
        writer.add_scalar('train/score', score, episode)
        writer.add_scalar('train/loss', loss.item(), episode)

        if episode % print_every == 0:
            recent_avg = np.mean(train_scores[-print_every:])
            print(
                f"Episode {episode:5d} | "
                f"Score {score:4d} | "
                f"Avg({print_every}) {recent_avg:6.1f} | "
                f"Steps {len(rewards):5d} | "
                f"Loss {loss.item():8.2f}"
            )
            writer.add_scalar('train/avg_score', recent_avg, episode)

        # ---- Periodic evaluation --------------------------------------
        if episode % eval_every == 0:
            # Build a deterministic policy function for evaluation:
            # pick the action with the highest probability (greedy).
            @torch.no_grad()
            def policy_fn(s, _policy=policy, _device=device):
                s_t = torch.tensor(s, dtype=torch.float32, device=_device)
                logits = _policy(s_t)
                return logits.argmax().item()

            eval_result = evaluate(policy_fn)
            eval_history.append((episode, eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], episode)
            print(
                f"  >> Eval @ {episode}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Final evaluation
    @torch.no_grad()
    def final_policy_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        logits = policy(s_t)
        return logits.argmax().item()

    final_eval = evaluate(final_policy_fn)
    print(
        f"\nFinal eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )

    writer.close()
    save_results('reinforce', train_scores, eval_result=final_eval)
    plot_training(
        train_scores,
        title='REINFORCE (Monte Carlo Policy Gradient)',
        path=os.path.join(os.path.dirname(__file__), 'results', 'reinforce.png'),
        eval_scores=eval_history,
    )

    return policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
