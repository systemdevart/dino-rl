"""
One-Step Actor-Critic for the Chrome Dino Game
===============================================
Sutton & Barto, Chapter 13.5 — Actor-Critic Methods

This file implements the one-step Actor-Critic algorithm, the most
fundamental member of the Actor-Critic family.  It is a natural evolution
from the REINFORCE algorithm and addresses its main weakness: high
variance in the gradient estimates.

Background — Why move beyond REINFORCE?
----------------------------------------
REINFORCE uses the *full* episode return G_t to weight the policy gradient.
Because G_t is a sum of many random rewards, its variance can be enormous,
causing noisy, unstable updates.  Actor-Critic methods fix this by
introducing a *Critic* — a learned value function V(s) — that provides a
low-variance, biased estimate of how good an action was.

The key ideas:
1. **Bootstrapping** (Sutton & Barto Ch 6, applied to policy gradients):
   Instead of waiting for the full return G_t, we use the *one-step TD
   target* r + gamma * V(s') as a proxy:

       delta = r + gamma * V(s') - V(s)        (TD error)

   delta is called the *advantage estimate* and replaces G_t in the
   policy gradient theorem.  It tells us: "Was this transition better or
   worse than what the critic expected?"

   - delta > 0  →  action was better than expected  →  increase its prob
   - delta < 0  →  action was worse than expected   →  decrease its prob

2. **Bias-variance trade-off**:
   Bootstrapping introduces *bias* because V(s') is an approximation,
   not the true value.  However, the *variance* drops dramatically
   because we no longer depend on all future rewards.  In practice, the
   variance reduction far outweighs the bias, leading to faster, more
   stable learning.

   REINFORCE:     high variance, zero bias    (uses G_t)
   Actor-Critic:  low variance, some bias     (uses r + gamma*V(s'))

3. **Online learning**:
   REINFORCE must wait until the end of an episode to compute returns.
   Actor-Critic updates the policy *at every single step*, using the
   instantaneous TD error.  This means:
   - Information propagates faster (no waiting for episode end)
   - Naturally handles continuing (non-episodic) tasks
   - Works with very long episodes where full returns are impractical

4. **Actor and Critic**:
   The algorithm maintains two components:
   - Actor (policy pi_theta):  decides which action to take
   - Critic (value  V_w):     estimates how good each state is

   In our neural network implementation these share a *feature extractor*
   (the hidden layers), saving computation and letting both benefit from
   the same learned state representation.  The last layer splits into:
   - Policy head → softmax probabilities (actor)
   - Value  head → scalar state value   (critic)

5. **Entropy bonus**:
   To encourage exploration, we add an entropy term to the actor loss:

       L_actor = -delta.detach() * log pi(a|s) - beta * H(pi(.|s))

   Entropy H is maximized when the policy is uniform (maximum
   uncertainty).  By rewarding high entropy, we discourage the policy
   from collapsing to a deterministic action too early, which helps
   avoid local optima.

Algorithm (One-Step Actor-Critic — episodic, Sutton & Barto p. 332):
---------------------------------------------------------------------
For each episode:
    s = env.reset()
    For each step:
        a ~ pi(a|s; theta)                    # sample from policy
        s', r, done = env.step(a)

        # Critic computes TD error:
        if done:
            delta = r - V(s; w)               # no future state
        else:
            delta = r + gamma * V(s'; w) - V(s; w)

        # Critic loss (minimize TD error squared):
        L_critic = 0.5 * delta^2

        # Actor loss (policy gradient with advantage = delta):
        # IMPORTANT: delta is detached — no gradient flows back through
        # the critic when updating the actor.  The critic is updated
        # only via L_critic.
        L_actor = -delta.detach() * log pi(a|s; theta) - beta * H(pi)

        L_total = L_actor + L_critic
        Backprop L_total, update theta and w

        s = s'
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from dino_rl.common import DinoFeatureEnv, evaluate, plot_training, save_results, create_writer, FEATURE_DIM, ACTION_SIZE, RESULTS_DIR

# ---------------------------------------------------------------------------
# Device detection — use GPU if available for faster training.
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Network Architecture: Shared Feature Extractor + Policy & Value Heads
# ============================================================================
#
# The Actor-Critic network uses a shared backbone:
#
#   state (8 features)
#       │
#       ▼
#   ┌───────────────┐
#   │ Linear(8, 128) │  ─ shared feature extractor
#   │     ReLU       │
#   │ Linear(128,128)│
#   │     ReLU       │
#   └───────┬───────┘
#           │
#     ┌─────┴──────┐
#     ▼            ▼
#  ┌────────┐  ┌────────┐
#  │ Policy │  │ Value  │
#  │ head   │  │ head   │
#  │ (128→2)│  │(128→1) │
#  │softmax │  │        │
#  └────────┘  └────────┘
#    Actor       Critic
#
# Sharing the feature extractor is efficient:  both actor and critic learn
# from the same representation, and the total parameter count stays small.
# The two heads specialize for their own tasks via separate output layers.
# ============================================================================

class ActorCriticNet(nn.Module):
    """
    Combined Actor-Critic network with a shared feature extractor.

    The actor outputs a probability distribution over actions (softmax).
    The critic outputs a scalar state-value estimate V(s).
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # --- Shared feature extractor (backbone) ---
        # Two hidden layers of 128 units with ReLU activations.
        # Both actor and critic build on this shared representation.
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # --- Actor head (policy) ---
        # Maps shared features → action logits → softmax probabilities.
        # We output raw logits and apply softmax in forward() so that
        # PyTorch's Categorical distribution can use them directly.
        self.policy_head = nn.Linear(128, action_dim)

        # --- Critic head (value function) ---
        # Maps shared features → single scalar V(s).
        # No activation: value can be any real number.
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            action_logits: raw logits for each action, shape (batch, action_dim)
            state_value:   scalar value estimate V(s), shape (batch, 1)
        """
        # Shared feature extraction
        features = self.shared(state)

        # Actor: produce action logits (raw, pre-softmax).
        # We do NOT apply softmax here; torch.distributions.Categorical
        # handles log-softmax internally for numerical stability.
        action_logits = self.policy_head(features)

        # Critic: produce state value estimate
        state_value = self.value_head(features)

        return action_logits, state_value


# ============================================================================
# Helper: Select action and return everything needed for the update
# ============================================================================

def select_action(network, state):
    """
    Given the current state, use the actor to sample an action and
    return all quantities needed for the loss computation.

    Returns:
        action (int):        the sampled action
        log_prob (Tensor):   log pi(a|s) — needed for the actor loss
        entropy (Tensor):    H(pi(.|s)) — the entropy bonus term
        value (Tensor):      V(s) — the critic's state-value estimate
    """
    # Convert state to tensor and add batch dimension
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    # Forward pass: get action logits and state value
    action_logits, state_value = network(state_tensor)

    # Create a categorical distribution from the action logits.
    # Using logits= is more numerically stable than passing softmax probs,
    # because Categorical uses log_softmax internally.
    dist = Categorical(logits=action_logits)

    # Sample an action from the distribution
    action = dist.sample()

    # log pi(a|s): the log-probability of the sampled action.
    # This is the core quantity in the policy gradient theorem:
    #   grad J ~ delta * grad log pi(a|s)
    log_prob = dist.log_prob(action)

    # Entropy H(pi) = -sum_a pi(a|s) * log pi(a|s)
    # Higher entropy = more exploration.  We will *add* this to the
    # objective (equivalently, subtract it from the loss) so that the
    # optimizer encourages higher-entropy policies.
    entropy = dist.entropy()

    return action.item(), log_prob, entropy, state_value.squeeze()


# ============================================================================
# Training Loop: One-Step Actor-Critic
# ============================================================================
#
# Hyperparameters:
#   lr             = 0.0005  — learning rate for Adam optimizer
#   gamma          = 0.99    — discount factor for future rewards
#   entropy_coeff  = 0.01    — weight of the entropy bonus
#
# The entropy coefficient controls the exploration-exploitation trade-off:
#   - Too high: policy stays near-uniform, agent explores aimlessly
#   - Too low:  policy becomes deterministic too quickly, may get stuck
#   - 0.01 is a reasonable default for discrete action spaces
# ============================================================================

def train(n_episodes=2000, lr=0.0005, gamma=0.99, entropy_coeff=0.01,
          print_every=10, eval_every=100, max_steps=25000, writer=None):
    """
    Train an Actor-Critic agent on the Dino environment.

    Args:
        n_episodes:    total training episodes
        lr:            learning rate for Adam
        gamma:         discount factor
        entropy_coeff: weight for the entropy bonus in the actor loss
        print_every:   print progress every N episodes
        eval_every:    run deterministic evaluation every N episodes
        max_steps:     maximum steps per episode (safety limit)

    Returns:
        network:       the trained ActorCriticNet
        train_scores:  list of per-episode training scores
        eval_history:  list of (episode, avg_eval_score) tuples
    """

    # --- Environment ---
    env = DinoFeatureEnv()

    # --- Network and optimizer ---
    network = ActorCriticNet(FEATURE_DIM, ACTION_SIZE).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    if writer is None:
        writer = create_writer('actor_critic')

    # --- Tracking ---
    train_scores = []
    eval_history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # ----- 1. Select action using the actor -----
            action, log_prob, entropy, value = select_action(network, state)

            # ----- 2. Take action in environment -----
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # ----- 3. Compute the TD error (delta) -----
            #
            # The TD error is the central quantity in Actor-Critic:
            #
            #   delta = r + gamma * V(s') - V(s)
            #
            # It measures the "surprise": how much better (or worse) the
            # actual transition was compared to the critic's prediction.
            #
            # At terminal states there is no future, so V(s') = 0:
            #   delta = r - V(s)
            #
            if done:
                # Terminal: no next state value
                td_target = torch.tensor(reward, dtype=torch.float32).to(device)
            else:
                # Non-terminal: bootstrap from V(s')
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    _, next_value = network(next_state_tensor)
                    next_value = next_value.squeeze()
                td_target = reward + gamma * next_value

            # TD error: delta = (r + gamma * V(s')) - V(s)
            # value is V(s) from the current step's forward pass
            delta = td_target - value

            # ----- 4. Compute losses -----
            #
            # ACTOR LOSS:
            #   L_actor = -delta.detach() * log pi(a|s) - entropy_coeff * H
            #
            # Why delta.detach()?
            #   The policy gradient theorem says we should weight log pi(a|s)
            #   by the *advantage* (here approximated by delta).  But delta
            #   depends on V(s), which has its own parameters.  If we let
            #   gradients flow through delta into V, we would be mixing the
            #   actor and critic objectives in a confusing way.
            #
            #   By detaching delta, we ensure:
            #   - The actor update only modifies the policy parameters
            #     (through log_prob and entropy)
            #   - The critic update only modifies the value parameters
            #     (through the critic loss below)
            #   This clean separation is important for training stability.
            #
            # The entropy term encourages exploration by penalizing
            # low-entropy (overly deterministic) policies.
            #
            actor_loss = -delta.detach() * log_prob - entropy_coeff * entropy

            # CRITIC LOSS:
            #   L_critic = 0.5 * delta^2
            #
            # This is just mean-squared TD error, encouraging V(s) to
            # accurately predict the (bootstrapped) return.
            # The 0.5 factor is conventional; it cancels the 2 from the
            # derivative of x^2, giving a cleaner gradient: grad = delta.
            #
            critic_loss = 0.5 * delta.pow(2)

            # COMBINED LOSS:
            # We sum actor and critic losses and do a single backward pass.
            # Because the feature extractor is shared, gradients from both
            # losses flow into it, training it to produce features useful
            # for both policy and value prediction.
            loss = actor_loss + critic_loss

            # ----- 5. Backprop and update -----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----- 6. Advance to next state -----
            state = next_state

            if done:
                break

        # --- Episode bookkeeping ---
        train_scores.append(info['score'])
        writer.add_scalar('train/score', info['score'], episode)

        if episode % print_every == 0:
            recent_avg = np.mean(train_scores[-print_every:])
            print(f"Episode {episode:5d} | "
                  f"Score: {info['score']:5d} | "
                  f"Avg({print_every}): {recent_avg:7.1f} | "
                  f"Steps: {step + 1:5d}")
            writer.add_scalar('train/avg_score', recent_avg, episode)

        # --- Periodic evaluation ---
        # Evaluate the current policy deterministically (argmax actions,
        # no domain randomization) to track true performance.
        if episode % eval_every == 0:
            network.eval()  # Set to eval mode (no effect here, but good practice)

            def policy_fn(s):
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                    logits, _ = network(s_tensor)
                    return logits.argmax(dim=-1).item()

            eval_result = evaluate(policy_fn, n_episodes=20)
            eval_history.append((episode, eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], episode)
            print(f"  >> Eval at episode {episode}: "
                  f"avg={eval_result['avg']:.1f}, "
                  f"min={eval_result['min']}, "
                  f"max={eval_result['max']}")

            network.train()  # Back to training mode

    # --- Final evaluation and save ---
    def final_policy_fn(s):
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            logits, _ = network(s_tensor)
            return logits.argmax(dim=-1).item()

    final_eval = evaluate(final_policy_fn, n_episodes=20)
    print(
        f"\nFinal eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )

    writer.close()
    save_results('actor_critic', train_scores, final_eval)
    plot_training(
        train_scores,
        title='One-Step Actor-Critic — Dino Game',
        path=os.path.join(RESULTS_DIR, 'actor_critic.png'),
        eval_scores=eval_history,
    )

    return network, train_scores, eval_history


# ============================================================================
# Main: Train, evaluate, save, and plot
# ============================================================================

if __name__ == '__main__':
    print(f"Device: {device}")
    print(f"Training One-Step Actor-Critic on Dino environment...")
    print(f"  State dim:  {FEATURE_DIM}")
    print(f"  Action dim: {ACTION_SIZE}")
    print("=" * 60)

    train(
        n_episodes=2000,
        lr=0.0005,
        gamma=0.99,
        entropy_coeff=0.01,
        print_every=10,
        eval_every=100,
    )
