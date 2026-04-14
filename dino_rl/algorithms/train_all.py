#!/usr/bin/env python3
"""
Train all RL algorithms sequentially with TensorBoard logging.

Usage:
    python -m dino_rl.algorithms.train_all              # train all algorithms
    python -m dino_rl.algorithms.train_all --algo ppo   # train just one

View TensorBoard:
    tensorboard --logdir results/runs
"""
import argparse
import time
import os

# Suppress TensorFlow warnings triggered by TensorBoard import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TENSORBOARD_BINARY'] = ''

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorboard')

from dino_rl.common import RESULTS_DIR, LOGS_DIR


def train_value_network():
    """Phase 2: Semi-gradient TD(0) value network."""
    from dino_rl.algorithms import value_network
    value_network.NUM_EPISODES = 1000
    value_network.EVAL_INTERVAL = 200
    value_network.EVAL_EPISODES = 10
    value_network.EPSILON_DECAY = 800
    from dino_rl.algorithms.value_network import train
    train()


def train_reinforce():
    """Phase 3: REINFORCE Monte Carlo policy gradient."""
    from dino_rl.algorithms.reinforce import train
    train(n_episodes=2000, print_every=50, eval_every=200)


def train_reinforce_baseline():
    """Phase 4: REINFORCE with baseline."""
    from dino_rl.algorithms.reinforce_baseline import train
    train(n_episodes=2000, print_every=50, eval_every=200)


def train_actor_critic():
    """Phase 5: One-step Actor-Critic."""
    from dino_rl.algorithms.actor_critic import train
    train(n_episodes=2000, print_every=50, eval_every=200)


def train_actor_critic_nstep():
    """Phase 6: N-step Actor-Critic."""
    from dino_rl.algorithms import actor_critic_nstep
    actor_critic_nstep.NUM_EPISODES = 2000
    actor_critic_nstep.EVAL_EVERY = 200
    actor_critic_nstep.PRINT_EVERY = 50
    from dino_rl.algorithms.actor_critic_nstep import train
    train()


def train_td_lambda():
    """Phase 7: True Online TD(lambda)."""
    from dino_rl.algorithms.td_lambda import train
    train(n_episodes=1000, print_every=50, eval_every=200)


def train_dqn_progression():
    """Phase 8: DQN stability pack (4 variants)."""
    from dino_rl.algorithms.dqn_progression import (
        VanillaDQN, DoubleDQN, DuelingDoubleDQN, DuelingDoubleDQN_PER,
        train_agent, evaluate, save_results, DEVICE
    )

    n_episodes = 500
    for AgentClass in [VanillaDQN, DoubleDQN, DuelingDoubleDQN, DuelingDoubleDQN_PER]:
        agent = AgentClass(DEVICE)
        scores, evals = train_agent(agent, n_episodes=n_episodes, eval_every=100)
        final_eval = evaluate(agent.policy, n_episodes=10)
        save_results(f'dqn_{agent.name}', scores, eval_result=final_eval)
        print(f"\n{agent.name} final eval: avg={final_eval['avg']:.1f}\n")


def train_ppo():
    """Phase 9: PPO."""
    from dino_rl.algorithms.ppo import train
    train(n_updates=5000, print_every=10, eval_every=50)


def train_a2c():
    """Phase 10: A2C."""
    from dino_rl.algorithms.a2c import train
    train(n_updates=3000, print_every=50, eval_every=500)


ALGORITHMS = {
    'value_network':       train_value_network,
    'reinforce':           train_reinforce,
    'reinforce_baseline':  train_reinforce_baseline,
    'actor_critic':        train_actor_critic,
    'actor_critic_nstep':  train_actor_critic_nstep,
    'td_lambda':           train_td_lambda,
    'dqn_progression':     train_dqn_progression,
    'ppo':                 train_ppo,
    'a2c':                 train_a2c,
}


def main():
    parser = argparse.ArgumentParser(description='Train RL algorithms with TensorBoard')
    parser.add_argument('--algo', type=str, default=None,
                        choices=list(ALGORITHMS.keys()),
                        help='Train a specific algorithm (default: all)')
    args = parser.parse_args()

    # Clear old TB runs
    runs_dir = LOGS_DIR
    if os.path.exists(runs_dir):
        import shutil
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir, exist_ok=True)

    # Also ensure results dir exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.algo:
        algos = {args.algo: ALGORITHMS[args.algo]}
    else:
        algos = ALGORITHMS

    print("=" * 70)
    print(f"  Training {len(algos)} algorithm(s) with TensorBoard logging")
    print(f"  View: tensorboard --logdir {runs_dir}")
    print("=" * 70)

    for name, train_fn in algos.items():
        print(f"\n{'='*70}")
        print(f"  Starting: {name}")
        print(f"{'='*70}\n")
        t0 = time.time()
        try:
            train_fn()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        print(f"\n  {name} completed in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("  All training complete!")
    print(f"  View results: tensorboard --logdir {runs_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
