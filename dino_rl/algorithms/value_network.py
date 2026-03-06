"""
Semi-gradient TD(0) with Neural Network Value Function -- Phase 2
=================================================================

Reference: Sutton & Barto, Ch 9.3 (Semi-gradient TD(0)), Ch 11 (The Deadly Triad)

This module implements a state-value function V(s) using a neural network,
trained with the semi-gradient TD(0) algorithm.  It is designed as an
educational companion to the Chrome Dino game RL project.


Background: from tabular to function approximation
---------------------------------------------------
In tabular TD(0) (Sutton & Barto Ch 6), we maintain a lookup table V[s] for
every state and update it with:

    V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

This works perfectly when the state space is small and discrete.  The Dino game
state, however, is an 8-dimensional continuous feature vector (obstacle
distance, speed, dino y-position, etc.), so we cannot enumerate all states.
Function approximation lets us generalize: instead of V[s], we learn a
parameterized function V(s; w) -- here a two-layer MLP -- that maps any
continuous state to a scalar value estimate.


Semi-gradient TD(0)
--------------------
The "semi-gradient" name (Sutton & Barto, Section 9.3) comes from a subtle but
critical detail in how we compute the gradient for the update.

The TD error is:

    delta = r + gamma * V(s'; w) - V(s; w)

If we naively differentiated the entire loss (delta^2) with respect to w, we
would get gradients flowing through *both* V(s; w) and V(s'; w).  That would be
the "full gradient" or "residual gradient" method, and it has well-known
problems: it converges to a different (less useful) fixed point and is much
slower in practice.

Instead, semi-gradient TD(0) treats the TD *target* -- r + gamma * V(s'; w) --
as a fixed constant when taking the gradient.  We only differentiate through
V(s; w):

    w <- w + alpha * delta * grad_w V(s; w)

In PyTorch terms, this means we call .detach() on the target:

    target = reward + gamma * V(next_state).detach()
    loss   = (V(state) - target) ** 2

The .detach() stops the gradient from flowing through V(s').  This is the same
idea behind the "target network" trick in DQN, though here we use the online
network for the target and simply stop the gradient.

Semi-gradient methods are *not* true gradient descent on any objective function
(the update direction is not the gradient of any single loss), but Sutton &
Barto prove convergence for linear function approximation under mild conditions
(Section 9.4).  For nonlinear approximation (neural networks), convergence is
not guaranteed in general, but works well in practice with appropriate care.


The Deadly Triad (Sutton & Barto, Chapter 11)
----------------------------------------------
Chapter 11 warns about the "deadly triad": the combination of three things that
can cause divergence:

    1. Function approximation  (generalizing across states)
    2. Bootstrapping           (using V(s') in the update target)
    3. Off-policy learning     (learning about a policy different from the one
                                generating the data)

Any two of these three are fine; it is the combination of all three that is
dangerous.  In this module:

    - We DO use function approximation (the neural network).
    - We DO bootstrap (the TD target uses V(s')).
    - We are ON-policy: the epsilon-greedy policy that generates experience is
      the same policy we are evaluating/improving.

Because we avoid the off-policy leg of the triad, the algorithm is relatively
stable.  DQN (Phase 1) uses all three legs -- it mitigates instability with a
replay buffer and a separate target network.  Understanding *why* those tricks
are necessary is a key educational takeaway.


Policy from V(s) alone -- the model-free control problem
---------------------------------------------------------
A value function V(s) tells us how good a state is, but not directly which
action to take.  To pick actions, we need either:

    (a) A model of the environment (to simulate s' for each candidate action
        and pick argmax_a V(s')), or
    (b) An action-value function Q(s, a) (as in DQN / SARSA).

Since we have neither a perfect model nor Q-values, we use a lightweight
heuristic: we maintain a running estimate of the average V(s') that followed
each action in recent experience.  The action whose recent successors had
higher average value is preferred (with epsilon-greedy exploration).

This is intentionally imperfect -- it demonstrates WHY a standalone value
function is insufficient for control without a model, and motivates the move
to Q-learning / policy gradient methods.  The V(s) we learn is still useful
for policy evaluation, critic baselines (Actor-Critic, Phase 4), and
understanding the TD learning mechanism itself.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dino_rl.common import (
    DinoFeatureEnv,
    evaluate,
    plot_training,
    save_results,
    FEATURE_DIM,
    ACTION_SIZE,
    create_writer,
    RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 0.001            # Learning rate for the value network optimizer
GAMMA = 0.99          # Discount factor -- how much we value future rewards
EPSILON_START = 1.0   # Initial exploration rate (fully random)
EPSILON_END = 0.01    # Final exploration rate after decay
EPSILON_DECAY = 2000  # Number of episodes over which epsilon decays linearly
NUM_EPISODES = 2000   # Total training episodes
EVAL_INTERVAL = 100   # Evaluate the policy every N episodes
EVAL_EPISODES = 20    # Number of episodes per evaluation run
MAX_STEPS = 25000     # Safety cap on steps per episode

# Running tracker window: how many recent (action, V(s')) pairs to remember
# for the simple action-selection heuristic.
ACTION_TRACKER_WINDOW = 500

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Value Network
# ===========================================================================
class ValueNetwork(nn.Module):
    """
    Two-layer MLP that estimates the state-value function V(s).

    Architecture:
        state (8,) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(1)

    The output is a single scalar: the estimated discounted return from state s
    under the current (implicit) policy.

    Compare with DQN, which outputs one value *per action* (Q(s, a) for all a).
    Here we output just one number, V(s), because the value function does not
    distinguish between actions -- it summarizes the overall goodness of a state.
    """

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # single scalar output: V(s)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: state tensor of shape (batch, FEATURE_DIM) or (FEATURE_DIM,).

        Returns:
            Tensor of shape (batch, 1) or (1,) -- the value estimate V(s).
        """
        return self.net(x)


# ===========================================================================
# Action-value tracker (heuristic for policy from V(s))
# ===========================================================================
class ActionValueTracker:
    """
    Lightweight tracker that records, for each action, the V(s') values that
    followed it in recent experience.

    This provides a crude action-selection signal:
        "Which action has historically led to higher-valued successor states?"

    It is NOT a proper Q-function -- it ignores the immediate reward and
    conflates the effects of different states.  It is included to illustrate
    the fundamental difficulty of control with V(s) alone:

        Without a model, V(s) cannot tell you which *action* is best.
        You need Q(s, a) or a model P(s' | s, a).

    Educational note:
        In Actor-Critic methods (Phase 4), we will replace this heuristic with
        a learned policy network (the "actor") that directly outputs action
        probabilities, while V(s) serves as the "critic" for variance reduction.
    """

    def __init__(self, n_actions: int = ACTION_SIZE, window: int = ACTION_TRACKER_WINDOW):
        # One deque per action, storing recent V(s') values.
        self.buffers = [deque(maxlen=window) for _ in range(n_actions)]

    def record(self, action: int, v_next: float):
        """Record that taking `action` was followed by a state with value `v_next`."""
        self.buffers[action].append(v_next)

    def best_action(self) -> int:
        """
        Return the action with the highest average V(s') in recent memory.

        Falls back to action 0 (do nothing) if we have no data yet, which is
        a safe default for the Dino game.
        """
        means = []
        for buf in self.buffers:
            if len(buf) == 0:
                means.append(-float("inf"))
            else:
                means.append(np.mean(buf))
        return int(np.argmax(means))


# ===========================================================================
# Epsilon schedule
# ===========================================================================
def get_epsilon(episode: int) -> float:
    """
    Linear epsilon decay from EPSILON_START to EPSILON_END over EPSILON_DECAY
    episodes, then constant at EPSILON_END.

    A linear schedule is simple and effective.  More sophisticated schedules
    (exponential, cosine annealing) exist but add complexity without changing
    the core algorithm.
    """
    if episode >= EPSILON_DECAY:
        return EPSILON_END
    return EPSILON_START + (EPSILON_END - EPSILON_START) * (episode / EPSILON_DECAY)


# ===========================================================================
# Action selection
# ===========================================================================
def select_action(
    tracker: ActionValueTracker,
    epsilon: float,
) -> int:
    """
    Epsilon-greedy action selection based on the action-value tracker.

    With probability epsilon, pick a random action (explore).
    Otherwise, pick the action whose recent successor states had the highest
    average V(s') (exploit).

    Note how different this is from DQN's action selection, which queries the
    Q-network directly: argmax_a Q(s, a).  Here we rely on a noisy historical
    average because V(s) alone does not provide per-action information.
    """
    if np.random.random() < epsilon:
        return np.random.randint(ACTION_SIZE)
    return tracker.best_action()


# ===========================================================================
# Training loop
# ===========================================================================
def train(writer=None):
    """
    Train a value network with semi-gradient TD(0) on the Dino game.

    Algorithm outline (per step within an episode):
        1. Observe state s, pick action a (epsilon-greedy via tracker).
        2. Execute a, observe reward r, next state s', done flag.
        3. Compute TD target:  y = r + gamma * V(s'; w)   [detached]
           (If done, y = r because there is no successor state.)
        4. Compute loss:       L = (V(s; w) - y)^2
        5. Backpropagate through V(s; w) ONLY (semi-gradient).
        6. Update weights w via Adam.
        7. Record (a, V(s')) in the action-value tracker.

    The semi-gradient nature is enforced by .detach() on the target.
    """
    print(f"Device: {device}")
    print(f"Training semi-gradient TD(0) value network for {NUM_EPISODES} episodes\n")

    # -- Environment ----------------------------------------------------------
    env = DinoFeatureEnv()

    # -- Network & optimizer --------------------------------------------------
    value_net = ValueNetwork(FEATURE_DIM).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=LR)

    if writer is None:
        writer = create_writer('value_network')

    # -- Action tracker -------------------------------------------------------
    tracker = ActionValueTracker()

    # -- Logging --------------------------------------------------------------
    train_scores = []        # score per training episode
    eval_log = []            # (episode, avg_eval_score) tuples

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        epsilon = get_epsilon(episode)

        episode_loss = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            steps += 1

            # 1. Select action ------------------------------------------------
            action = select_action(tracker, epsilon)

            # 2. Environment step ---------------------------------------------
            next_state, reward, done, info = env.step(action)

            # 3. Convert to tensors -------------------------------------------
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)       # (1, 8)
            ns_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device) # (1, 8)

            # 4. Compute V(s) and the TD target --------------------------------
            v_s = value_net(s_tensor)           # (1, 1) -- gradient flows here

            with torch.no_grad():
                # SEMI-GRADIENT: V(s') is treated as a constant.
                # This is the defining characteristic of semi-gradient TD(0).
                # We do NOT backpropagate through V(s') -- only through V(s).
                #
                # Why?  The TD target  r + gamma * V(s')  is our *estimate* of
                # the true value.  We treat it like a supervised-learning label
                # that happens to move as training progresses.  Differentiating
                # through the target would give the "residual gradient" method,
                # which converges to a different (generally worse) fixed point
                # and exhibits slower learning (Baird, 1995).
                v_ns = value_net(ns_tensor)     # (1, 1) -- detached

            if done:
                # Terminal state: no future reward.
                td_target = torch.FloatTensor([[reward]]).to(device)
            else:
                td_target = reward + GAMMA * v_ns  # already detached

            # 5. Loss and semi-gradient update ---------------------------------
            #
            # L = (V(s; w) - y)^2     where y = r + gamma * V(s'; w_detached)
            #
            # grad_w L = 2 * (V(s; w) - y) * grad_w V(s; w)
            #
            # This is *not* the gradient of any single objective function
            # (because y depends on w through V(s'; w), which we froze).
            # Sutton & Barto call this a "semi-gradient" method.  It is biased
            # but has lower variance than Monte Carlo, and converges for linear
            # approximation.  For nonlinear (neural net) approximation,
            # convergence is empirical but widely observed in practice.
            loss = nn.functional.mse_loss(v_s, td_target)

            optimizer.zero_grad()
            loss.backward()           # gradients flow through v_s only
            optimizer.step()

            episode_loss += loss.item()

            # 6. Update action tracker ----------------------------------------
            # Record what V(s') was after taking this action.
            # This feeds the crude action-selection heuristic.
            tracker.record(action, v_ns.item())

            # 7. Advance state ------------------------------------------------
            state = next_state

            if done:
                break

        # -- Episode bookkeeping ----------------------------------------------
        score = info["score"]
        train_scores.append(score)
        writer.add_scalar('train/score', score, episode)

        if episode % 10 == 0:
            avg_loss = episode_loss / max(steps, 1)
            recent_avg = np.mean(train_scores[-50:]) if len(train_scores) >= 50 else np.mean(train_scores)
            print(
                f"Episode {episode:5d} | "
                f"Score {score:6d} | "
                f"Avg(50) {recent_avg:7.1f} | "
                f"Loss {avg_loss:.4f} | "
                f"Eps {epsilon:.3f} | "
                f"Steps {steps}"
            )
            writer.add_scalar('train/avg_loss', avg_loss, episode)
            writer.add_scalar('train/epsilon', epsilon, episode)
            writer.add_scalar('train/avg_score_50', recent_avg, episode)

        # -- Periodic evaluation ----------------------------------------------
        if episode % EVAL_INTERVAL == 0:
            # Build a deterministic policy from the current tracker + value net.
            # For evaluation we use epsilon=0 (pure greedy via tracker).
            def _make_policy(net, trk):
                """Closure to capture current network and tracker state."""
                def policy_fn(state_np):
                    return trk.best_action()
                return policy_fn

            policy = _make_policy(value_net, tracker)
            eval_result = evaluate(policy, n_episodes=EVAL_EPISODES)
            eval_log.append((episode, eval_result["avg"]))
            writer.add_scalar('eval/avg_score', eval_result['avg'], episode)
            writer.add_scalar('eval/min_score', eval_result['min'], episode)
            writer.add_scalar('eval/max_score', eval_result['max'], episode)
            print(
                f"  >> Eval @ {episode}: "
                f"avg={eval_result['avg']:.1f}  "
                f"min={eval_result['min']}  "
                f"max={eval_result['max']}"
            )

    # -- Final evaluation -----------------------------------------------------
    print("\n--- Final Evaluation ---")

    def final_policy(state_np):
        return tracker.best_action()

    final_eval = evaluate(final_policy, n_episodes=EVAL_EPISODES)
    print(
        f"Final eval: avg={final_eval['avg']:.1f}  "
        f"min={final_eval['min']}  max={final_eval['max']}"
    )

    # -- Save results and plot ------------------------------------------------
    save_results("value_network_td0", train_scores, eval_result=final_eval)

    plot_training(
        train_scores,
        title="Semi-gradient TD(0) Value Network -- Training Scores",
        path=os.path.join(RESULTS_DIR, "value_network_td0_training.png"),
        eval_scores=eval_log,
    )

    writer.close()

    print("\nDone.")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    train()
