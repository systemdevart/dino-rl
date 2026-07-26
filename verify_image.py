"""Formal image-DQN verification in independent deterministic browsers.

Usage: python verify_image.py <checkpoint> [episodes] [cap]
Pass bar: min episode score >= 5000. Defaults retain the formal 8x22000
protocol used to promote the frozen image checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from dino_rl.browser_env import ChromeDinoImageEnv
from image_dqn import DuelingCNNQ, greedy_action_indices


FORMAL_EPISODES = 8
FORMAL_STEP_CAP = 22000
LEGACY_STATE_SHAPE = (4, 84, 336)
LEGACY_ENCODER = "impala"
LEGACY_ACTION_SIZE = 3
LEGACY_MASK_SCORE = False


def resolve_device(requested="auto"):
    """Resolve a portable verifier device instead of requiring CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _inferred_action_size(model_state):
    for name in ("advantage.bias", "module.advantage.bias"):
        if name in model_state:
            return int(model_state[name].shape[0])
    return LEGACY_ACTION_SIZE


def image_policy_spec(checkpoint):
    """Return the persisted policy contract, with legacy-checkpoint defaults."""
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    model_state = _model_state(checkpoint)
    state_shape = tuple(metadata.get("state_shape", LEGACY_STATE_SHAPE))
    encoder = metadata.get("encoder", LEGACY_ENCODER)
    action_size = int(metadata.get(
        "action_size", _inferred_action_size(model_state)
    ))
    no_duck = bool(metadata.get("no_duck", False))
    mask_score = bool(metadata.get("mask_score", LEGACY_MASK_SCORE))
    action_names = tuple(metadata.get(
        "action_names", ("noop", "jump", "duck")[:action_size]
    ))

    if len(state_shape) != 3 or any(int(dim) <= 0 for dim in state_shape):
        raise ValueError(f"invalid image state_shape: {state_shape!r}")
    if encoder not in {"impala", "nature"}:
        raise ValueError(f"unsupported image encoder: {encoder!r}")
    if action_size not in {2, 3}:
        raise ValueError(f"browser image policy needs 2 or 3 actions, got {action_size}")
    if len(action_names) != action_size:
        raise ValueError("checkpoint action_names do not match action_size")
    if no_duck and action_size != 3:
        raise ValueError("no_duck metadata requires a three-action Q head")
    observation_mode = metadata.get("observation_mode", "image")
    if observation_mode != "image":
        raise ValueError(
            f"checkpoint observation_mode must be 'image', got {observation_mode!r}"
        )
    return {
        "state_shape": state_shape,
        "encoder": encoder,
        "action_size": action_size,
        "action_names": action_names,
        "no_duck": no_duck,
        "mask_score": mask_score,
        "model_state_dict": model_state,
    }


def load_image_policy(checkpoint_path, requested_device="auto"):
    device = resolve_device(requested_device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    spec = image_policy_spec(checkpoint)
    net = DuelingCNNQ(
        spec["state_shape"], spec["action_size"], spec["encoder"]
    ).to(device)
    net.load_state_dict(spec["model_state_dict"])
    net.eval()
    return net, device, spec


@torch.no_grad()
def policy_action(net, obs, device, no_duck=False):
    q_values = net(
        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    )
    return int(greedy_action_indices(q_values, no_duck=no_duck).item())


def run_episode(
    episode_index,
    net,
    device,
    spec,
    cap,
    env_factory=ChromeDinoImageEnv,
):
    channels, height, width = spec["state_shape"]
    for attempt in range(3):
        env = None
        try:
            env = env_factory(
                obs_size=height,
                obs_width=width,
                frame_stack=channels,
                deterministic=True,
                frames_per_action=1,
                reward_mode="survival",
                mask_score=spec["mask_score"],
                headless=True,
            )
            obs = env.reset()
            if tuple(obs.shape) != spec["state_shape"]:
                raise ValueError(
                    f"browser observation shape {tuple(obs.shape)} does not "
                    f"match checkpoint {spec['state_shape']}"
                )
            done = False
            recovered = False
            steps = 0
            score = 0
            while not done and steps < cap:
                action = policy_action(
                    net, obs, device, no_duck=spec["no_duck"]
                )
                obs, _reward, done, info = env.step(action)
                recovered = bool(info.get("browser_recovered", False))
                score = int(info.get("score", score))
                steps += 1
                if recovered:
                    break
            if not recovered:
                return score, steps
            print(
                f"episode {episode_index}: browser recovery, retry "
                f"{attempt + 1}/3",
                flush=True,
            )
        finally:
            if env is not None:
                env.close()
    raise RuntimeError(
        f"episode {episode_index}: browser failed on all recovery attempts"
    )


def verify(checkpoint_path, episodes, cap, requested_device="auto"):
    if episodes < 1 or cap < 1:
        raise ValueError("episodes and cap must be positive")
    net, device, spec = load_image_policy(checkpoint_path, requested_device)
    print(
        f"policy: device={device} encoder={spec['encoder']} "
        f"shape={spec['state_shape']} actions={spec['action_names']} "
        f"no_duck={spec['no_duck']} mask_score={spec['mask_score']}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(4, episodes)) as executor:
        results = list(executor.map(
            lambda index: run_episode(index, net, device, spec, cap),
            range(episodes),
        ))
    scores = np.array([result[0] for result in results])
    cap_survivals = sum(1 for result in results if result[1] >= cap)
    print(
        f">>> VERIFY {checkpoint_path} n={episodes} cap={cap}: "
        f"mean={scores.mean():.0f} min={scores.min()} max={scores.max()} "
        f">=5000: {int(np.sum(scores >= 5000))}/{episodes} "
        f"cap-survivals={cap_survivals}"
    )
    print("scores:", sorted(scores.tolist()))
    return scores, cap_survivals


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", nargs="?", type=int, default=FORMAL_EPISODES)
    parser.add_argument("cap", nargs="?", type=int, default=FORMAL_STEP_CAP)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (default: CUDA when available, otherwise CPU)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    verify(args.checkpoint, args.episodes, args.cap, args.device)


if __name__ == "__main__":
    main()
