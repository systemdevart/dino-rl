"""Evaluate a saved DQN or PPO policy in the chosen environment backend."""

import argparse

from dino_rl.common import evaluate
from dino_rl.policy_loader import load_policy
from dino_rl.policy_paths import default_policy_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Dino policy in simulation or browser"
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
        "--max-steps", type=int, default=50000, help="Max steps per episode"
    )
    parser.add_argument("--device", default="cpu", help="Torch device for inference")
    parser.add_argument(
        "--env-backend",
        choices=("sim", "browser"),
        default="sim",
        help="Environment backend to evaluate on",
    )
    parser.add_argument(
        "--browser-url", default="chrome://dino", help="Browser backend page URL"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show Chrome instead of headless mode when using browser backend",
    )
    parser.add_argument(
        "--browser-accelerate",
        action="store_true",
        help="Use real Dino acceleration in browser backend",
    )
    args = parser.parse_args()

    policy = load_policy(args.model, algo=args.algo, device=args.device)
    env_kwargs = {}
    if args.env_backend == "browser":
        env_kwargs = {
            "browser_headless": not args.show_browser,
            "browser_accelerate": args.browser_accelerate,
            "browser_url": args.browser_url,
        }
    result = evaluate(
        policy.act,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        env_backend=args.env_backend,
        env_kwargs=env_kwargs,
    )

    print(
        f"Eval {policy.algo.upper()} | episodes={args.episodes} | "
        f"avg={result['avg']:.1f} min={result['min']} max={result['max']}"
    )
    print(f"Scores: {result['scores']}")


if __name__ == "__main__":
    main()
