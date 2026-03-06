"""
A2C -- Advantage Actor-Critic (Synchronous)
============================================
Mnih et al., 2016: "Asynchronous Methods for Deep Reinforcement Learning"

This file implements A2C, the *synchronous* variant of A3C.  The original
paper (Mnih 2016) proposed A3C -- Asynchronous Advantage Actor-Critic --
where multiple worker threads each run their own copy of the environment
and asynchronously push gradient updates to a shared parameter server.
The asynchrony was motivated by two goals:

    1.  **Decorrelated samples.**  On-policy methods like REINFORCE suffer
        when consecutive transitions are highly correlated (the agent sees
        the same stretch of gameplay many times in a row).  Running many
        environments in parallel -- each in a different state -- produces
        a diverse mini-batch of experience per update, which stabilises
        learning in the same way that experience replay stabilises DQN,
        but without needing a replay buffer (important because on-policy
        methods cannot reuse old data).

    2.  **Wall-clock speed.**  Multiple workers collect data faster.

It turned out, however, that the *asynchrony* is not essential and can
actually hurt reproducibility (non-deterministic gradient ordering).  A2C
simply runs N environments *synchronously* -- step all N envs, collect a
batch of transitions, compute one gradient, update.  This is simpler to
implement, easier to debug, and performs just as well (or better) than
A3C in practice.  OpenAI's baselines library uses A2C for this reason.

How A2C differs from REINFORCE
------------------------------
REINFORCE (implemented in reinforce.py) uses *complete episodes* and the
*actual return* G_t as the gradient weight.  This is an unbiased but
high-variance estimator because G_t depends on many future random actions.

A2C improves on this in three ways:

    1.  **Bootstrapping (n-step returns).**  Instead of waiting until the
        end of the episode, we collect only n_steps transitions and then
        *bootstrap* the remaining future value using a learned value
        function V(s):

            R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
                  + gamma^n * V(s_{t+n})

        This introduces some bias (V is approximate) but dramatically
        reduces variance.  The parameter n_steps controls the bias-variance
        trade-off: n=1 is pure TD (low variance, high bias); n=infinity
        is Monte Carlo (zero bias, high variance).  n=5 is a common sweet
        spot.

    2.  **Advantage baseline.**  The gradient weight is the *advantage*:

            A_t = R_t - V(s_t)

        Subtracting the value baseline does not change the expected
        gradient (it is a constant w.r.t. the action), but it centres the
        signal: actions better than average get positive advantage, worse
        get negative.  This dramatically reduces variance compared to raw
        returns.

    3.  **Entropy bonus.**  We add an entropy term to the loss:

            L_entropy = - sum_t sum_a pi(a|s_t) log pi(a|s_t)

        Maximising entropy prevents the policy from collapsing to a
        deterministic one too early, which encourages exploration.  The
        coefficient (entropy_coeff) controls how much we value exploration
        vs. exploitation.

Architecture: Actor-Critic with shared trunk
---------------------------------------------
The network has a *shared* feature-extraction body (two hidden layers)
that feeds into two separate *heads*:

    state -> [Linear(128) -> ReLU -> Linear(128) -> ReLU]  (shared trunk)
                |                                    |
                v                                    v
         policy head (actor)                value head (critic)
         Linear -> softmax                  Linear -> scalar
         pi(a|s)                            V(s)

Sharing the trunk means the policy and value function learn common
features, which is more parameter-efficient and often improves learning.

Comparison with PPO
-------------------
PPO (Proximal Policy Optimisation, Schulman 2017) is a direct successor
of A2C that adds a *clipped surrogate objective*:

    L_clip = min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t )

where r_t = pi_new(a|s) / pi_old(a|s) is the probability ratio.  This
clipping prevents the policy from changing too much in a single update,
which makes training more stable.

A2C has *no such safeguard*.  The policy can change arbitrarily in one
step, which occasionally causes performance collapses.  In exchange,
A2C is simpler (no need to store old log-probs or compute ratios) and
has fewer hyperparameters to tune.  For many problems A2C works well
enough, and it serves as the conceptual foundation for PPO.

N-step return computation
-------------------------
For each environment, we collect a trajectory of n_steps transitions:

    s_0, a_0, r_0, s_1, a_1, r_1, ..., s_{n-1}, a_{n-1}, r_{n-1}, s_n

The n-step return for timestep t is:

    R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1-t} * r_{n-1}
          + gamma^{n-t} * V(s_n)      [if the episode did not end]

If the episode terminated at some step k <= n-1, then rewards after step k
are zero and there is no bootstrap term (the value of a terminal state is
zero by definition).

We compute this by iterating backwards from step n-1:

    R_{n-1} = r_{n-1} + gamma * V(s_n) * (1 - done_{n-1})
    R_t     = r_t + gamma * R_{t+1} * (1 - done_t)

The (1 - done) term zeroes out the bootstrap whenever the episode ended,
correctly handling terminal states.
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
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network for A2C.

    Architecture:
        state (8 features)
            -> Linear(128) -> ReLU       (shared)
            -> Linear(128) -> ReLU       (shared)
            |                       |
            v                       v
        Linear(ACTION_SIZE)     Linear(1)
        [policy logits]        [state value]

    The shared trunk learns features useful for both predicting the policy
    (which action to take) and the value (how good is this state).  This
    is more parameter-efficient than having two separate networks, and the
    shared representation often improves learning since the value function
    gradient also shapes the feature layers.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Policy head (actor): outputs logits for each action
        # These get passed through softmax (via Categorical) to produce pi(a|s)
        self.policy_head = nn.Linear(128, action_dim)
        # Value head (critic): outputs a single scalar V(s)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the shared trunk and both heads.

        Args:
            x: state tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            policy_logits: shape (batch, action_dim) -- raw logits (pre-softmax)
            value: shape (batch, 1) -- estimated state value V(s)
        """
        features = self.shared(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


# ---------------------------------------------------------------------------
# N-step return computation
# ---------------------------------------------------------------------------

def compute_nstep_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute n-step returns for a batch of parallel environments.

    This implements the backward recursion:

        R_{n-1} = r_{n-1} + gamma * V(s_n) * (1 - done_{n-1})
        R_t     = r_t     + gamma * R_{t+1} * (1 - done_t)

    The (1 - done) mask is critical: when an episode terminates at step t,
    the return should not include any bootstrap from beyond that episode.
    The value of a terminal state is zero, so we simply cut off the recursion.

    Args:
        rewards:  shape (n_steps, n_envs) -- rewards collected at each step
        dones:    shape (n_steps, n_envs) -- True if episode ended at that step
        bootstrap_value: shape (n_envs,) -- V(s_n) for bootstrapping the final step
        gamma: discount factor

    Returns:
        returns: shape (n_steps, n_envs) tensor of n-step returns
    """
    n_steps = rewards.shape[0]
    returns = np.zeros_like(rewards, dtype=np.float32)

    # Start from the bootstrap value V(s_n) and work backwards.
    # We detach the bootstrap value because we do not want gradients to
    # flow through the target (same as in DQN where we detach the target).
    R = bootstrap_value.detach().cpu().numpy()

    for t in reversed(range(n_steps)):
        # If done[t] is True, the episode ended after receiving reward[t].
        # The return should just be the accumulated reward without bootstrap.
        # The (1 - dones[t]) term handles this elegantly.
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R

    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_updates: int = 2000,
    n_envs: int = 8,
    n_steps: int = 5,
    gamma: float = 0.99,
    lr: float = 0.0005,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    print_every: int = 10,
    eval_every: int = 100,
    device: str | None = None,
    writer=None,
):
    """
    Train a policy using A2C on the Dino game with parallel environments.

    The training loop structure:

        for each update iteration:
            1. Collect n_steps transitions from each of the n_envs environments.
               This produces a batch of (n_steps * n_envs) = 40 transitions.
            2. Compute n-step returns for each environment (backward recursion).
            3. Compute advantages: A_t = R_t - V(s_t).
            4. Compute the three loss components:
                 - Policy loss:  -mean( A_t.detach() * log pi(a_t | s_t) )
                   We detach the advantage so that the policy gradient does
                   not try to "cheat" by changing the value estimate.
                 - Value loss:   mean( (R_t - V(s_t))^2 )
                   Standard MSE regression toward the n-step return target.
                 - Entropy:      -mean( sum_a pi(a|s) * log pi(a|s) )
                   Added as a bonus (subtracted from total loss) to encourage
                   exploration.
            5. Total loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            6. Single backprop + gradient step.

    Why parallel environments help:
        Each environment is in a different state (different obstacle positions,
        different speeds).  A batch containing transitions from 8 independent
        environments is much more diverse than 8 consecutive transitions from
        a single environment.  This decorrelation stabilises the gradient
        estimate and is the key insight behind A3C/A2C.

    Args:
        n_updates: number of parameter update iterations
        n_envs: number of parallel environments
        n_steps: number of steps to collect from each env per update
        gamma: discount factor
        lr: learning rate for Adam optimiser
        entropy_coeff: weight for the entropy bonus
        value_coeff: weight for the value loss
        print_every: print progress every N updates
        eval_every: run deterministic evaluation every N updates
        device: 'cuda' or 'cpu'; auto-detected if None
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_steps = n_updates * n_steps * n_envs
    print(
        f"A2C  |  device={device}  lr={lr}  gamma={gamma}  "
        f"n_envs={n_envs}  n_steps={n_steps}\n"
        f"       entropy_coeff={entropy_coeff}  value_coeff={value_coeff}\n"
        f"       {n_updates} updates x {n_steps * n_envs} transitions = "
        f"{total_steps} total env steps"
    )

    # Create N parallel environments.
    # Each environment runs independently -- they do not communicate.
    # We maintain a current state for each environment and step them
    # all together in a vectorised fashion.
    envs = [DinoFeatureEnv() for _ in range(n_envs)]
    current_states = np.array([env.reset() for env in envs], dtype=np.float32)
    # current_states shape: (n_envs, FEATURE_DIM)

    model = ActorCritic(FEATURE_DIM, ACTION_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if writer is None:
        writer = create_writer('a2c')

    # Track episode scores from all environments
    train_scores: list[int] = []
    eval_history: list[tuple[int, float]] = []  # (update_step, avg_score)

    # Running episode score tracker for each env (accumulated score within
    # the current episode; reset when the env is done).
    episode_scores = [0] * n_envs

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------
    for update in range(1, n_updates + 1):
        # Storage for the n_steps transitions from all envs.
        # We store things in (n_steps, n_envs) arrays for easy batch processing.
        mb_states = np.zeros((n_steps, n_envs, FEATURE_DIM), dtype=np.float32)
        mb_actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        mb_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        mb_dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        mb_log_probs = []   # list of tensors, one per step
        mb_values = []      # list of tensors, one per step

        # ---- 1. Collect n_steps transitions from each environment -----
        for step in range(n_steps):
            mb_states[step] = current_states

            # Forward pass through the network for all envs at once.
            # This is much more efficient than calling forward() separately
            # for each environment, because it uses batched matrix multiplies.
            states_t = torch.tensor(current_states, dtype=torch.float32, device=device)
            with torch.no_grad():
                # We use no_grad for action selection because we will
                # re-compute log_probs with gradients in the loss computation.
                # Actually, for A2C we need gradients through the log_probs
                # and values.  But it is cleaner to collect transitions
                # first and then do a single forward pass for the loss.
                # However, for simplicity, we compute log_probs here with
                # gradients enabled so we can use them directly.
                pass

            # Re-do forward pass with gradients enabled
            logits, values = model(states_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()                  # shape: (n_envs,)
            log_probs = dist.log_prob(actions)       # shape: (n_envs,)

            mb_actions[step] = actions.cpu().numpy()
            mb_log_probs.append(log_probs)
            mb_values.append(values.squeeze(-1))     # shape: (n_envs,)

            # Step all environments
            for i, env in enumerate(envs):
                next_state, reward, done, info = env.step(actions[i].item())
                mb_rewards[step, i] = reward
                mb_dones[step, i] = float(done)
                episode_scores[i] += 1  # track steps as proxy; use score

                if done:
                    # Episode finished in this env -- record the score
                    train_scores.append(info['score'])
                    episode_scores[i] = 0
                    # Reset the environment and use the new initial state
                    next_state = env.reset()

                current_states[i] = next_state

        # ---- 2. Compute bootstrap value V(s_n) for the final state ----
        # After collecting n_steps, we need V(s_n) to bootstrap the return.
        # s_n is the current state of each env (the state *after* the last
        # collected transition).
        with torch.no_grad():
            final_states_t = torch.tensor(
                current_states, dtype=torch.float32, device=device
            )
            _, bootstrap_values = model(final_states_t)
            bootstrap_values = bootstrap_values.squeeze(-1)  # shape: (n_envs,)

        # ---- 3. Compute n-step returns and advantages -----------------
        returns = compute_nstep_returns(mb_rewards, mb_dones, bootstrap_values, gamma)
        returns = returns.to(device)  # shape: (n_steps, n_envs)

        # Stack the per-step values and log_probs into tensors.
        values_t = torch.stack(mb_values)        # shape: (n_steps, n_envs)
        log_probs_t = torch.stack(mb_log_probs)  # shape: (n_steps, n_envs)

        # Advantage = n-step return - predicted value
        # We detach the returns because they are *targets* (like in DQN).
        # The advantage used in the policy gradient is also detached from
        # the value function parameters, so the policy gradient does not
        # inadvertently affect the value function through the advantage.
        advantages = returns.detach() - values_t.detach()

        # ---- 4. Compute losses ----------------------------------------
        #
        # Policy loss (actor):
        #   L_policy = - (1/B) * sum A_t * log pi(a_t | s_t)
        #
        # We detach advantages so that the policy gradient does not flow
        # back through the value function.  The gradient of L_policy w.r.t.
        # policy parameters theta is:
        #
        #   nabla L_policy = - (1/B) * sum A_t * nabla log pi(a_t | s_t)
        #
        # which is exactly the advantage actor-critic policy gradient.
        policy_loss = -(advantages * log_probs_t).mean()

        # Value loss (critic):
        #   L_value = (1/B) * sum (R_t - V(s_t))^2
        #
        # This is standard mean squared error regression.  The value
        # function learns to predict the n-step return.
        value_loss = (returns.detach() - values_t).pow(2).mean()

        # Entropy bonus:
        #   H(pi) = - sum_a pi(a|s) * log pi(a|s)
        #
        # We compute this by rebuilding the distributions from the stored
        # logits.  Higher entropy means the policy is more spread out
        # across actions, which encourages exploration.
        #
        # We need to recompute the distributions to get the entropy.
        # Since mb_states is available, we do a single batched forward pass.
        all_states_t = torch.tensor(
            mb_states.reshape(-1, FEATURE_DIM), dtype=torch.float32, device=device
        )
        all_logits, _ = model(all_states_t)
        all_dist = Categorical(logits=all_logits)
        entropy = all_dist.entropy().mean()

        # Total loss combines all three components:
        #   L = L_policy + value_coeff * L_value - entropy_coeff * H
        #
        # Note the signs:
        # - L_policy already has a negative sign (we minimise negative reward)
        # - L_value is always positive (MSE), and we minimise it
        # - Entropy is subtracted because we want to MAXIMISE it (more exploration)
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        # ---- 5. Gradient step -----------------------------------------
        optimizer.zero_grad()
        total_loss.backward()
        # Optional: gradient clipping to prevent exploding gradients.
        # A2C can sometimes produce large gradients, especially early in
        # training when the value function is inaccurate.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # ---- Bookkeeping ---------------------------------------------
        if update % print_every == 0 and len(train_scores) > 0:
            recent_n = min(50, len(train_scores))
            recent_avg = np.mean(train_scores[-recent_n:])
            total_env_steps = update * n_steps * n_envs
            print(
                f"Update {update:5d} | "
                f"Steps {total_env_steps:7d} | "
                f"Episodes {len(train_scores):5d} | "
                f"Avg({recent_n}) {recent_avg:6.1f} | "
                f"Policy {policy_loss.item():7.3f} | "
                f"Value {value_loss.item():7.3f} | "
                f"Entropy {entropy.item():5.3f}"
            )
            writer.add_scalar('train/avg_score', recent_avg, update)
            writer.add_scalar('train/policy_loss', policy_loss.item(), update)
            writer.add_scalar('train/value_loss', value_loss.item(), update)
            writer.add_scalar('train/entropy', entropy.item(), update)

        # ---- Periodic evaluation --------------------------------------
        if update % eval_every == 0:
            @torch.no_grad()
            def policy_fn(s, _model=model, _device=device):
                s_t = torch.tensor(s, dtype=torch.float32, device=_device)
                logits, _ = _model(s_t)
                return logits.argmax().item()

            eval_result = evaluate(policy_fn)
            # Use the equivalent "episode number" as the number of completed
            # training episodes so far, for the eval_history x-axis.
            eval_history.append((len(train_scores), eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], update)
            print(
                f"  >> Eval @ update {update}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    @torch.no_grad()
    def final_policy_fn(s):
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        logits, _ = model(s_t)
        return logits.argmax().item()

    final_eval = evaluate(final_policy_fn)
    print(
        f"\nFinal eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )

    writer.close()
    save_results('a2c', train_scores, eval_result=final_eval)
    plot_training(
        train_scores,
        title='A2C (Advantage Actor-Critic)',
        path=os.path.join(os.path.dirname(__file__), 'results', 'a2c.png'),
        eval_scores=eval_history,
    )

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
