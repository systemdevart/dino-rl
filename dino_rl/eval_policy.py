"""Evaluate a saved DQN or PPO policy in the simulator."""

import argparse

from dino_rl.common import evaluate
from dino_rl.policy_loader import load_policy
from dino_rl.policy_paths import default_policy_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Dino policy in simulation"
    )
    parser.add_argument(
        "--model",
        default=default_policy_path(),
        help=(
            "Checkpoint path (defaults to checkpoints/dino_ppo_best.pth when "
            "present, otherwise checkpoints/dino_runner.pth)"
        ),
    )
    parser.add_argument(
        "--algo",
        choices=("auto", "dqn", "ppo"),
        default="auto",
        help="Checkpoint type (default: auto-detect)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of eval episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=25000, help="Max steps per episode"
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device for inference"
    )
    args = parser.parse_args()

    policy = load_policy(args.model, algo=args.algo, device=args.device)
    result = evaluate(
        policy.act, n_episodes=args.episodes, max_steps=args.max_steps
    )

    print(
        f"Eval {policy.algo.upper()} | episodes={args.episodes} | "
        f"avg={result['avg']:.1f} min={result['min']} max={result['max']}"
    )
    print(f"Scores: {result['scores']}")


if __name__ == '__main__':
    main()
