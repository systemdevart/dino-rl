"""
N-Step Actor-Critic for the Chrome Dino Game
=============================================

**Phase 6** of the educational RL algorithms series.

Sutton & Barto references:
    - Chapter 7.1 : N-step TD Prediction
    - Chapter 13.5 : Actor-Critic Methods

Overview
--------
The one-step actor-critic (TD(0) actor-critic) bootstraps after every single
step: it observes (s_t, a_t, r_t, s_{t+1}), uses V(s_{t+1}) as the bootstrap
target, and immediately updates.  This gives *low variance* (because the
update depends on a single reward sample) but *high bias* (because V(s_{t+1})
is an approximation, not the true value).

At the other extreme, REINFORCE (Monte Carlo policy gradient) waits until the
episode ends and uses the full return G_t = r_t + gamma*r_{t+1} + ... as the
target.  This is *unbiased* (G_t is an unbiased sample of the true value) but
has *high variance* (every future reward contributes noise).

N-step methods interpolate between these two extremes:

    G_t^{(n)} = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
                + gamma^n * V(s_{t+n})

    - When n = 1 : we recover TD(0) (one-step actor-critic)
    - When n -> infinity (or episode ends before n steps) : we recover MC

Bias-Variance Tradeoff
----------------------
    n=1  : lowest variance, highest bias   (heavy bootstrapping)
    n=5  : moderate variance, moderate bias (sweet spot for many tasks)
    n=20 : higher variance, lower bias     (closer to Monte Carlo)
    n=inf: highest variance, zero bias     (pure Monte Carlo)

Choosing n is a hyperparameter tuning problem.  In practice, n=5 or n=20 are
common defaults.  For the Dino game (short episodes, fast dynamics), n=5
provides a good balance: the agent sees enough future consequences of its
actions to learn "jump timing" without accumulating excessive noise.

Architecture
------------
Shared MLP backbone (128-128 hidden layers) with two separate heads:

    features (8) -> Linear(128) -> ReLU -> Linear(128) -> ReLU
                                                        |
                                            +-----------+-----------+
                                            |                       |
                                      Policy head              Value head
                                      Linear(2)                Linear(1)
                                      Softmax                  (scalar)

Weight sharing between policy and value is beneficial because both need to
understand the same game state; the shared layers learn general state
representations while the heads specialise.

Algorithm (per episode)
-----------------------
1. Collect n transitions: (s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_{n-1}, a_{n-1}, r_{n-1})
   (or fewer if the episode ends before n steps).
2. Compute the n-step return for each timestep t in the collected batch:
       If the episode ended at step k < n:
           G_t = r_t + gamma*r_{t+1} + ... + gamma^{k-t-1}*r_{k-1}
       Else:
           G_t = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n * V(s_{t+n})
3. Compute the advantage for each step: A_t = G_t - V(s_t)
4. Losses:
       Actor loss  = -sum[ log pi(a_t | s_t) * A_t ]  (policy gradient with advantage)
       Critic loss = sum[ (G_t - V(s_t))^2 ]          (regression on value)
       Entropy     = -sum[ pi * log pi ]               (exploration bonus, subtracted)
       Total loss  = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy
5. Backprop and update.

The key difference from one-step AC is that we collect n steps before updating,
and the returns incorporate multiple real rewards before bootstrapping.  This
lets the credit assignment "see further" without waiting for the entire episode.

Why n-step returns help in the Dino game
-----------------------------------------
The Dino game has delayed consequences: the decision to jump must be made
several frames *before* the obstacle arrives.  With n=1, the agent only sees
the immediate +0.1 survival reward and must rely entirely on the value function
to propagate information backwards.  With n=5, the return already includes
the next 5 rewards, so if the agent crashes at step t+3, the n-step return
at step t will directly incorporate the -1.0 crash penalty (discounted).
This makes credit assignment faster and more reliable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from dino_rl.common import DinoFeatureEnv, evaluate, plot_training, save_results, create_writer, FEATURE_DIM, ACTION_SIZE, RESULTS_DIR

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# Prefer CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
# For a small network like ours the difference is negligible, but this makes
# the code portable to any machine.
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 0.0005             # Learning rate for Adam
GAMMA = 0.99            # Discount factor
N_STEPS = 5             # Number of look-ahead steps for n-step returns
ENTROPY_COEFF = 0.01    # Entropy bonus coefficient (encourages exploration)
NUM_EPISODES = 2000     # Total training episodes
PRINT_EVERY = 10        # Print training progress every N episodes
EVAL_EVERY = 100        # Run deterministic evaluation every N episodes
MAX_STEPS = 25000       # Safety cap on episode length


# ---------------------------------------------------------------------------
# Network: Shared backbone with Policy (actor) and Value (critic) heads
# ---------------------------------------------------------------------------
class ActorCriticNet(nn.Module):
    """
    Shared-parameter actor-critic network.

    The shared backbone learns a compressed state representation.  The policy
    head outputs a probability distribution over actions (softmax), and the
    value head outputs a scalar estimate of V(s).

    Architecture:
        Input(8) -> FC(128, ReLU) -> FC(128, ReLU) -> policy_head(2) + value_head(1)
    """

    def __init__(self, state_dim: int = FEATURE_DIM, action_dim: int = ACTION_SIZE):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy head: outputs action logits (unnormalised log-probabilities)
        # Categorical distribution will apply softmax internally.
        self.policy_head = nn.Linear(128, action_dim)

        # Value head: outputs a single scalar V(s)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Args:
            state: tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            action_logits: (batch, action_dim) -- raw logits for Categorical
            state_value:   (batch, 1)          -- estimated V(s)
        """
        features = self.shared(state)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features)
        return action_logits, state_value


# ---------------------------------------------------------------------------
# N-Step Actor-Critic Agent
# ---------------------------------------------------------------------------
class NStepActorCritic:
    """
    N-Step Advantage Actor-Critic agent.

    The agent interacts with the environment in chunks of n steps.  After each
    chunk it computes n-step returns and performs a single gradient update.

    The n-step return for step t (within a chunk of length T) is:

        If the episode ended at step k <= T:
            G_t = sum_{i=t}^{k-1} gamma^{i-t} * r_i          (no bootstrap)
        Else:
            G_t = sum_{i=t}^{T-1} gamma^{i-t} * r_i + gamma^{T-t} * V(s_T)

    We compute these efficiently by iterating backwards from the last step.
    """

    def __init__(self, lr=LR, gamma=GAMMA, n_steps=N_STEPS,
                 entropy_coeff=ENTROPY_COEFF):
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff

        self.net = ActorCriticNet().to(device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=lr)

    # ----- action selection ------------------------------------------------
    def select_action(self, state: np.ndarray):
        """
        Sample an action from the policy and return it along with the log-prob.

        Args:
            state: numpy array of shape (FEATURE_DIM,)

        Returns:
            action:   int -- sampled action
            log_prob: tensor -- log pi(a|s), retained for the policy gradient
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, 8)
        logits, _ = self.net(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(0)

    def get_value(self, state: np.ndarray) -> torch.Tensor:
        """Compute V(s) for a single state (no gradient needed for bootstrap)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.net(state_t)
        return value.squeeze()  # scalar tensor

    # ----- n-step return computation ---------------------------------------
    @staticmethod
    def compute_nstep_returns(rewards, dones, bootstrap_value, gamma):
        """
        Compute n-step returns by iterating backwards through the collected
        transitions.

        Given rewards [r_0, r_1, ..., r_{T-1}] and a bootstrap value V(s_T),
        the return for the *last* step is:

            G_{T-1} = r_{T-1}                         (if done at T)
                    = r_{T-1} + gamma * V(s_T)         (if not done)

        Then working backwards:

            G_t = r_t + gamma * G_{t+1}               (if not done at t+1)
                = r_t                                  (if done at t)

        Wait -- this looks like standard TD(lambda=1) returns, not n-step.
        The key is that T (the chunk length) is *at most* n, so each G_t
        naturally includes at most n real rewards before the bootstrap kicks
        in.  If the episode terminated inside the chunk, the returns become
        partial Monte Carlo returns (no bootstrapping after termination).

        This backward recursion is numerically equivalent to the explicit
        forward summation:
            G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k} + gamma^{T-t} * V(s_T) * (1-done)

        but is simpler to implement and more numerically stable.

        Args:
            rewards:         list of T floats
            dones:           list of T booleans (True if episode ended at that step)
            bootstrap_value: scalar float, V(s_T) (0 if episode ended)
            gamma:           discount factor

        Returns:
            list of T floats -- the n-step return for each step in the chunk
        """
        T = len(rewards)
        returns = [0.0] * T

        # Start from the bootstrap value at the boundary
        G = bootstrap_value
        for t in reversed(range(T)):
            if dones[t]:
                # Episode terminated at step t.  The return is just the
                # terminal reward (no future rewards or bootstrapping).
                G = rewards[t]
            else:
                # Standard Bellman recursion with real reward + discounted future
                G = rewards[t] + gamma * G
            returns[t] = G

        return returns

    # ----- training loop for one episode -----------------------------------
    def train_episode(self, env, max_steps=MAX_STEPS):
        """
        Train on a single episode using n-step returns.

        The episode is divided into chunks of at most n steps.  After each
        chunk we compute returns and update the network.  This is sometimes
        called "n-step semi-gradient" updating.

        Returns:
            total_reward: float  -- undiscounted sum of rewards for the episode
            score:        int    -- game score at episode end
        """
        state = env.reset()
        total_reward = 0.0
        score = 0
        step_count = 0

        done = False
        while not done and step_count < max_steps:
            # ---- Collect up to n steps of experience ----
            #
            # We store: states, log_probs (for the policy gradient),
            # rewards, dones, and state values (for the critic loss).
            states = []
            log_probs = []
            rewards = []
            dones_chunk = []
            values = []

            chunk_start_state = state

            for _ in range(self.n_steps):
                # Get value estimate for current state (need gradient for critic)
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, value = self.net(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Store transition data
                states.append(state_t)
                log_probs.append(log_prob.squeeze(0))
                values.append(value.squeeze())

                # Step the environment
                next_state, reward, done, info = env.step(action.item())
                rewards.append(reward)
                dones_chunk.append(done)
                score = info['score']
                total_reward += reward
                step_count += 1

                state = next_state

                if done:
                    break

            # ---- Compute bootstrap value for the n-step return ----
            #
            # If the episode did NOT end in this chunk, we bootstrap from
            # V(s_{t+n}) -- the value of the state we landed in after n steps.
            # If it DID end (done=True), the bootstrap is 0 because there is
            # no future reward after a terminal state.
            if done:
                bootstrap_value = 0.0
            else:
                bootstrap_value = self.get_value(state).item()

            # ---- Compute n-step returns for every step in the chunk ----
            returns = self.compute_nstep_returns(
                rewards, dones_chunk, bootstrap_value, self.gamma
            )

            # ---- Compute losses and update ----
            #
            # For each step t in the chunk:
            #   advantage  A_t = G_t - V(s_t)          (n-step advantage)
            #   actor loss     = -log pi(a_t|s_t) * A_t (REINFORCE-style PG)
            #   critic loss    = (G_t - V(s_t))^2       (MSE regression)
            #   entropy        = -sum_a pi(a|s_t) * log pi(a|s_t)
            #
            # Total loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy
            #
            # The 0.5 factor on the critic loss is a common convention that
            # keeps the critic gradient magnitude comparable to the actor.
            # The entropy term encourages exploration by penalising overly
            # confident (low-entropy) policies.

            returns_t = torch.FloatTensor(returns).to(device)
            values_t = torch.stack(values)           # (T,)
            log_probs_t = torch.stack(log_probs)     # (T,)

            # Advantages: stop gradient through returns (they are fixed targets)
            advantages = returns_t - values_t.detach()

            # Actor loss: negative log-prob weighted by advantage
            actor_loss = -(log_probs_t * advantages).sum()

            # Critic loss: MSE between predicted values and n-step returns
            critic_loss = ((returns_t - values_t) ** 2).sum()

            # Entropy bonus: recompute distributions for the stored states
            # We need the full distribution (not just the sampled log-prob)
            # to compute the entropy H = -sum_a pi(a) * log pi(a).
            states_batch = torch.cat(states, dim=0)  # (T, 8)
            logits_batch, _ = self.net(states_batch)
            dist_batch = Categorical(logits=logits_batch)
            entropy = dist_batch.entropy().sum()

            # Combined loss
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

            # Gradient step
            self.optimiser.zero_grad()
            loss.backward()
            # Gradient clipping prevents catastrophically large updates that
            # can destabilise training (common in RL due to high variance).
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimiser.step()

        return total_reward, score

    # ----- deterministic policy for evaluation -----------------------------
    def policy(self, state: np.ndarray) -> int:
        """
        Deterministic policy: pick the action with the highest probability.
        Used during evaluation (no exploration).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = self.net(state_t)
        return logits.argmax(dim=-1).item()


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------
def train(writer=None):
    """
    Main training loop.

    Trains the N-step Actor-Critic agent for NUM_EPISODES episodes.
    Periodically prints progress and runs deterministic evaluation.
    At the end, saves results and plots the training curve.
    """
    print("=" * 60)
    print("N-Step Actor-Critic  (Sutton & Barto Ch 7.1 + Ch 13.5)")
    print("=" * 60)
    print(f"Device           : {device}")
    print(f"Learning rate    : {LR}")
    print(f"Discount (gamma) : {GAMMA}")
    print(f"N steps          : {N_STEPS}")
    print(f"Entropy coeff    : {ENTROPY_COEFF}")
    print(f"Episodes         : {NUM_EPISODES}")
    print("=" * 60)

    env = DinoFeatureEnv()
    agent = NStepActorCritic(
        lr=LR, gamma=GAMMA, n_steps=N_STEPS, entropy_coeff=ENTROPY_COEFF
    )

    if writer is None:
        writer = create_writer('actor_critic_nstep')

    train_scores = []
    eval_history = []  # list of (episode, avg_eval_score)
    recent_scores = []

    for episode in range(1, NUM_EPISODES + 1):
        total_reward, score = agent.train_episode(env)
        train_scores.append(score)
        writer.add_scalar('train/score', score, episode)
        recent_scores.append(score)

        # ---- Periodic progress printing ----
        if episode % PRINT_EVERY == 0:
            avg = np.mean(recent_scores[-PRINT_EVERY:])
            print(f"Episode {episode:5d} | "
                  f"Score {score:5d} | "
                  f"Avg(last {PRINT_EVERY}) {avg:7.1f} | "
                  f"Reward {total_reward:7.1f}")
            writer.add_scalar('train/avg_score', avg, episode)

        # ---- Periodic deterministic evaluation ----
        if episode % EVAL_EVERY == 0:
            eval_result = evaluate(agent.policy, n_episodes=20)
            eval_history.append((episode, eval_result['avg']))
            writer.add_scalar('eval/avg_score', eval_result['avg'], episode)
            print(f"  >> Eval @ ep {episode}: "
                  f"avg={eval_result['avg']:.1f}  "
                  f"min={eval_result['min']}  "
                  f"max={eval_result['max']}")

    # ---- Final evaluation ----
    print("\n" + "=" * 60)
    print("Final evaluation (20 episodes, deterministic policy)")
    print("=" * 60)
    final_eval = evaluate(agent.policy, n_episodes=20)
    print(f"  Avg: {final_eval['avg']:.1f}  "
          f"Min: {final_eval['min']}  "
          f"Max: {final_eval['max']}")

    # ---- Save results and training curve ----
    writer.close()
    save_results('actor_critic_nstep', train_scores, eval_result=final_eval)
    plot_training(
        train_scores,
        title='N-Step Actor-Critic (n=5) Training Curve',
        path=os.path.join(RESULTS_DIR, 'actor_critic_nstep.png'),
        eval_scores=eval_history,
    )

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    train()
