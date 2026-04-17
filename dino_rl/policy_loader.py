"""Load trained DQN or PPO policies behind one inference interface."""

from dataclasses import dataclass, field

import numpy as np
import torch

from dino_rl.algorithms.ppo import ActorCritic
from dino_rl.common import ACTION_SIZE
from dino_rl.feature_contract import FEATURE_DIM
from dino_rl.networks import DuelingDQN


def _detect_algo(checkpoint: dict) -> str:
    algo = checkpoint.get("algo")
    if algo in {"dqn", "ppo"}:
        return algo
    if "model_state_dict" in checkpoint:
        return "ppo"
    if "target_model" in checkpoint or "model" in checkpoint:
        return "dqn"
    raise ValueError(
        "Unable to detect checkpoint type. Pass --algo explicitly or use a "
        "checkpoint saved by this project."
    )


@dataclass
class LoadedPolicy:
    """Thin wrapper that exposes a single deterministic act() method."""

    algo: str
    model: torch.nn.Module
    device: torch.device
    checkpoint: dict = field(repr=False)

    def act(self, features: np.ndarray | list[float]) -> int:
        feature_vec = np.asarray(features, dtype=np.float32).reshape(-1)
        expected_dim = int(
            self.checkpoint.get(
                "feature_dim",
                getattr(self.model, "state_dim", FEATURE_DIM),
            )
        )
        if feature_vec.shape[0] > expected_dim:
            feature_vec = feature_vec[:expected_dim]
        elif feature_vec.shape[0] < expected_dim:
            feature_vec = np.pad(
                feature_vec,
                (0, expected_dim - feature_vec.shape[0]),
                mode="constant",
            )

        state = torch.as_tensor(
            feature_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if self.algo == "ppo":
                logits, _ = self.model(state)
                return logits.argmax(dim=1).item()
            q_values = self.model(state)
            return q_values.argmax(dim=1).item()


def load_policy(
    weight_path: str,
    algo: str = "auto",
    device: str = "cpu",
) -> LoadedPolicy:
    """Load a checkpoint and expose it through a uniform policy wrapper."""
    torch_device = torch.device(device)
    checkpoint = torch.load(weight_path, map_location=torch_device, weights_only=True)

    if algo == "auto":
        algo = _detect_algo(checkpoint)

    if algo == "ppo":
        model = ActorCritic(
            checkpoint.get("feature_dim", FEATURE_DIM),
            checkpoint.get("action_size", ACTION_SIZE),
            checkpoint.get("latent_dim", 128),
        ).to(torch_device)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError("PPO checkpoint missing 'model_state_dict'.")
    elif algo == "dqn":
        model = DuelingDQN(
            checkpoint.get("feature_dim", FEATURE_DIM),
            checkpoint.get("action_size", ACTION_SIZE),
        ).to(torch_device)
        state_dict = checkpoint.get("model")
        if state_dict is None:
            raise ValueError("DQN checkpoint missing 'model'.")
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'.")

    model.load_state_dict(state_dict, strict=(algo != "ppo"))
    model.eval()
    return LoadedPolicy(
        algo=algo, model=model, device=torch_device, checkpoint=checkpoint
    )
