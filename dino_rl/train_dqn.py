"""
DQN Agent for Chrome Dinosaur Game.

Uses Dueling Double DQN with experience replay and soft target updates.
Trains on a pure Python simulation of the Chrome dino game.

Key fixes from original:
- np.argmax -> np.max in Q-learning update (critical bug fix)
- Double DQN for reduced overestimation bias
- Dueling architecture for better value estimation
- Feature-based observation (distance to obstacle, dino height, etc.)
  instead of raw pixels for faster convergence
- Soft target updates (polyak averaging) for stable Q-values
- Batch training on GPU
- Train every N steps during gameplay (not just at episode end)
- Periodic eval episodes with no exploration
- Switched from TensorFlow/Keras to PyTorch for better compatibility
"""
import numpy as np
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from dino_rl.env import DinoRunEnv
from dino_rl.networks import DuelingDQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FEATURE_DIM = 8  # Number of features from env.get_features() (includes speed)


class Agent:
    def __init__(self, action_size: int):
        self.weight_backup = os.path.join(
            os.path.dirname(__file__), '..', 'checkpoints', 'dino_runner.pth'
        )
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.gamma = 0.99
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0003
        self.tau = 0.005  # Soft target update rate
        self.train_freq = 4  # Train every 4 steps
        self.min_replay_size = 2000  # Min transitions before training starts

        os.makedirs(os.path.dirname(self.weight_backup), exist_ok=True)

        self.model = DuelingDQN(FEATURE_DIM, action_size).to(device)
        self.target_model = DuelingDQN(FEATURE_DIM, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        if os.path.isfile(self.weight_backup):
            checkpoint = torch.load(self.weight_backup, map_location=device,
                                    weights_only=True)
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Loaded weights from {self.weight_backup} (eps={self.epsilon:.4f})")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def soft_update_target(self):
        """Polyak averaging: target = tau * model + (1-tau) * target."""
        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save_model(self):
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, self.weight_backup)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_t)
            return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch], dtype=np.float32)
        next_states = np.array([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch], dtype=np.float32)

        states_t = torch.FloatTensor(states).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online model selects action, target model evaluates
            best_actions = self.model(next_states_t).argmax(dim=1)
            next_q = self.target_model(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TRexRunner:
    """Training loop for the DQN dino agent."""

    def __init__(self):
        self.batch_size = 256
        self.episodes = 10000
        self.eval_freq = 25  # Run eval episode every N training episodes
        self.eval_runs = 5   # Average over N eval episodes for reliability
        # Training env: no DR (browser game always uses dino_x=50)
        self.env = DinoRunEnv(domain_randomization=False, feature_noise=0.0,
                              skip_clear_time=True)
        # Eval env uses no randomization for consistent benchmarking
        self.eval_env = DinoRunEnv(domain_randomization=False, feature_noise=0.0,
                                   skip_clear_time=True)
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.action_size)

    def run_eval_episode(self):
        """Run eval_runs episodes with epsilon=0, return average score."""
        scores = []
        total_steps = 0
        for _ in range(self.eval_runs):
            self.eval_env.reset()
            state = self.eval_env.get_features()
            game_score = 0
            for t in range(25000):
                action = self.agent.act(state, eval_mode=True)
                _, _, score, done = self.eval_env.step(action)
                state = self.eval_env.get_features()
                game_score = score
                if done:
                    break
            scores.append(game_score)
            total_steps += t + 1
        return int(np.mean(scores)), total_steps // self.eval_runs, min(scores)

    def run(self):
        total_time = 0
        best_score = 0
        best_eval = 0
        scores_window = deque(maxlen=100)

        total_params = sum(p.numel() for p in self.agent.model.parameters())
        print(f"Network size: {total_params:,} parameters")

        try:
            for e in range(self.episodes):
                self.env.reset()
                state = self.env.get_features()

                game_score = 0

                for t in range(25000):
                    total_time += 1

                    action = self.agent.act(state)
                    _, reward, score, done = self.env.step(action)

                    # Reward shaping (matching actor-critic's proven config):
                    # +0.01 per step keeps discounted survival reward small,
                    # -10.0 crash penalty dominates for short (bad) episodes.
                    if done:
                        reward = -10.0
                    else:
                        reward = 0.01

                    next_state = self.env.get_features()

                    self.agent.remember(state, action, reward, next_state, done)

                    # Train and soft-update target
                    if (total_time % self.agent.train_freq == 0 and
                            len(self.agent.memory) >= self.agent.min_replay_size):
                        self.agent.replay(self.batch_size)
                        self.agent.soft_update_target()

                    state = next_state
                    game_score = score

                    if done:
                        break

                self.agent.decay_epsilon()

                scores_window.append(game_score)
                avg_score = np.mean(scores_window)

                if game_score > best_score:
                    best_score = game_score

                print(
                    f"Ep {e+1:4d}/{self.episodes} | "
                    f"Score: {game_score:5d} | Best: {best_score:5d} | "
                    f"Avg100: {avg_score:7.1f} | "
                    f"Eps: {self.agent.epsilon:.4f} | "
                    f"Steps: {t+1:5d} | Mem: {len(self.agent.memory):6d}"
                )

                # Periodic eval with no exploration
                if (e + 1) % self.eval_freq == 0:
                    eval_score, eval_steps, eval_min = self.run_eval_episode()
                    print(
                        f"  ** EVAL ({self.eval_runs} runs): "
                        f"Avg {eval_score:5d} | Min {eval_min:5d} | "
                        f"Steps {eval_steps:5d} | "
                        f"Best eval: {best_eval:5d}"
                    )
                    if eval_score > best_eval:
                        best_eval = eval_score
                        self.agent.save_model()
                        print(f"  ** New best eval! Saved model.")

                    if best_eval >= 10000:
                        print(f"\n*** TARGET REACHED! Eval avg: {best_eval} ***")
                        break

                if (e + 1) % 100 == 0:
                    self.agent.save_model()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print(f"\nFinal save... Best train: {best_score}, Best eval: {best_eval}")
            self.agent.save_model()


def main():
    dino = TRexRunner()
    dino.run()


if __name__ == '__main__':
    main()
