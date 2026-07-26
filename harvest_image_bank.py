"""Harvest a provenance-stamped image bank from a corrected feature teacher.

The promoted corrected feature-DQN drives the game from the paired feature
stream while this script records image observations. Exploratory transitions
are retained for diagnostics but marked non-expert so the DQfD loader excludes
them from its protected demonstration buffer.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from dino_rl.browser_env import VecChromeDinoImageEnv
from dino_rl.feature_contract import FEATURE_DIM
from dino_rl.networks import DuelingDQN
from dino_rl.train_dqn import greedy_feature_actions


DEMO_BANK_FORMAT = "dino_image_demo_bank_v1"
DEFAULT_TEACHER = "checkpoints/dino_browser_dqn_speeddropm_candidate.pth"
DEFAULT_OUTPUT = "checkpoints/image_dqn_bank_clean.npz"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--image-width", type=int, default=336)
    parser.add_argument(
        "--target", type=int, default=60000, help="transitions to bank"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="exploration rate; exploratory rows are marked non-expert",
    )
    parser.add_argument(
        "--speed-curriculum",
        type=float,
        default=0.0,
        help="fraction of episodes starting at a random speed in [8,13]",
    )
    parser.add_argument(
        "--milestone-bonus",
        type=float,
        default=0.0,
        help="reward added when score crosses each 100-point milestone",
    )
    parser.add_argument("--crash-penalty", type=float, default=-10.0)
    parser.add_argument("--alive-reward", type=float, default=0.01)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seconds", type=int, default=5400)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser


def load_teacher(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = (
        metadata.get("model", metadata.get("model_state_dict", checkpoint))
        if isinstance(checkpoint, dict) else checkpoint
    )
    teacher = DuelingDQN(FEATURE_DIM, 3).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    action_mode = {
        "no_duck": bool(metadata.get("no_duck", False)),
    }
    provenance = {
        "kind": "corrected_browser_feature_dqn_checkpoint",
        "path": path,
        "sha256": sha256_file(path),
        "action_mode": action_mode,
        "checkpoint_env_backend": metadata.get("env_backend"),
        "checkpoint_observation_mode": metadata.get("observation_mode"),
        "teacher_or_demo_data": False,
    }
    return teacher, action_mode, provenance


def build_bank_metadata(args, teacher_provenance, actions, dones, expert):
    return {
        "format": DEMO_BANK_FORMAT,
        "kind": "browser_image_demo_bank",
        "created_by": "harvest_image_bank.py",
        "teacher_or_demo_data": True,
        "teacher": teacher_provenance,
        "policy_observation": "engineered_features",
        "recorded_observation": "grayscale_frame_stack",
        "state_shape": (4, args.image_size, args.image_width),
        "environment": {
            "backend": "chrome_browser",
            "deterministic": True,
            "frames_per_action": 1,
            "normal_start_probability": 1.0 - args.speed_curriculum,
            "speed_curriculum_probability": args.speed_curriculum,
        },
        "collection": {
            "seed": args.seed,
            "requested_transitions": args.target,
            "actual_transitions": int(len(actions)),
            "epsilon": args.epsilon,
            "expert_transitions": int(np.sum(expert)),
            "terminal_transitions": int(np.sum(dones)),
            "action_counts": np.bincount(actions, minlength=3).tolist(),
            "alive_reward": args.alive_reward,
            "crash_penalty": args.crash_penalty,
            "milestone_bonus": args.milestone_bonus,
        },
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, teacher_action_mode, teacher_provenance = load_teacher(
        args.teacher, device
    )
    print(f"teacher: {args.teacher}", flush=True)

    @torch.no_grad()
    def teacher_actions(feature_rows):
        actions = np.zeros(len(feature_rows), dtype=np.int64)
        valid = [
            (index, features)
            for index, features in enumerate(feature_rows)
            if features is not None and len(features)
        ]
        if valid:
            states = torch.tensor(
                np.array([features for _, features in valid]),
                dtype=torch.float32,
                device=device,
            )
            selected = greedy_feature_actions(
                teacher(states),
                **teacher_action_mode,
            ).cpu().numpy()
            for selected_index, (row_index, _) in enumerate(valid):
                actions[row_index] = selected[selected_index]
        exploratory = np.random.rand(len(feature_rows)) < args.epsilon
        actions[exploratory] = np.random.randint(
            0, 3, int(exploratory.sum())
        )
        return actions, ~exploratory

    env = VecChromeDinoImageEnv(
        args.num_envs,
        obs_size=args.image_size,
        obs_width=args.image_width,
        reward_mode="survival",
        mask_score=False,
        deterministic=True,
        frames_per_action=1,
        start_speed_prob=args.speed_curriculum,
    )
    try:
        obs = env.reset()
        bank_obs, bank_actions, bank_rewards = [], [], []
        bank_next_obs, bank_dones, bank_expert = [], [], []
        episode_scores = []
        milestones = np.zeros(args.num_envs, dtype=np.int64)
        start = time.time()
        while (
            len(bank_actions) < args.target
            and time.time() - start < args.max_seconds
        ):
            feature_rows = env.get_feature_states()
            actions, expert = teacher_actions(feature_rows)
            next_obs, _env_reward, dones, infos = env.step(actions)
            for index in range(args.num_envs):
                if infos[index].get("browser_recovered", False):
                    milestones[index] = 0
                    continue
                reward = (
                    args.crash_penalty if dones[index] else args.alive_reward
                )
                score = int(infos[index].get("score", 0))
                if dones[index]:
                    milestones[index] = 0
                elif args.milestone_bonus:
                    milestone = score // 100
                    if milestone > milestones[index]:
                        reward += args.milestone_bonus * (
                            milestone - milestones[index]
                        )
                        milestones[index] = milestone
                bank_obs.append((obs[index] * 255).astype(np.uint8))
                bank_actions.append(int(actions[index]))
                bank_rewards.append(np.float32(reward))
                bank_next_obs.append(
                    (next_obs[index] * 255).astype(np.uint8)
                )
                bank_dones.append(
                    np.float32(1.0 if dones[index] else 0.0)
                )
                bank_expert.append(bool(expert[index]))
                if dones[index]:
                    episode_scores.append(score)
            obs = next_obs
            if len(bank_actions) % 6000 < args.num_envs:
                elapsed = time.time() - start
                steps_per_second = len(bank_actions) / max(1e-9, elapsed)
                print(
                    f"[{int(elapsed)}s] bank={len(bank_actions)}/{args.target} "
                    f"({steps_per_second:.0f} tr/s) "
                    f"crashes={len(episode_scores)}",
                    flush=True,
                )
    finally:
        env.close()

    actions = np.array(bank_actions, dtype=np.int64)
    dones = np.array(bank_dones, dtype=np.float32)
    expert = np.array(bank_expert, dtype=bool)
    counts = np.bincount(actions, minlength=3)
    print(
        f"BANK: {len(actions)} transitions | "
        f"actions noop/jump/duck={counts.tolist()} "
        f"| terminal={int(np.sum(dones))} | expert={int(np.sum(expert))}",
        flush=True,
    )
    metadata = build_bank_metadata(
        args, teacher_provenance, actions, dones, expert
    )
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.array(bank_obs, dtype=np.uint8),
        actions=actions,
        rewards=np.array(bank_rewards, dtype=np.float32),
        next_obs=np.array(bank_next_obs, dtype=np.uint8),
        dones=dones,
        expert=expert,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
