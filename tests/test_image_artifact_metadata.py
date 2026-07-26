import argparse
import inspect
import json
import tempfile
import unittest

import numpy as np
import torch

from dino_rl.browser_env import ChromeDinoImageEnv
from dino_rl.networks import DuelingDQN
from harvest_image_bank import (
    DEFAULT_OUTPUT,
    DEFAULT_TEACHER,
    build_bank_metadata,
    build_parser as build_harvest_parser,
    load_teacher,
)
from image_dqn import (
    DuelingCNNQ,
    IMAGE_ACTION_NAMES,
    IMAGE_CHECKPOINT_FORMAT,
    demo_bank_metadata,
    image_checkpoint_metadata,
    save_checkpoint,
    validate_replay_snapshot,
    validate_resume_request,
)
from verify_image import image_policy_spec, load_image_policy, policy_action


class ImageArtifactMetadataTest(unittest.TestCase):
    def test_image_dqn_opts_out_of_historical_score_mask(self):
        constructor_default = inspect.signature(
            ChromeDinoImageEnv.__init__
        ).parameters["mask_score"].default
        self.assertIs(constructor_default, True)

        args = argparse.Namespace(
            encoder="impala",
            no_duck=False,
            no_duck_margin=0.01,
            deterministic=True,
            frames_per_action=1,
            init_from="",
        )
        metadata = image_checkpoint_metadata(args, (4, 84, 336))
        spec = image_policy_spec({
            "model_state_dict": {"advantage.bias": torch.zeros(3)},
            **metadata,
        })

        self.assertIs(metadata["mask_score"], False)
        self.assertIs(spec["mask_score"], False)

    def test_new_checkpoint_round_trip_is_self_describing(self):
        model = DuelingCNNQ((4, 16, 32), 3, "impala")
        config = {"frames_per_action": 1, "seed": 17}
        provenance = {
            "kind": "browser_image_dqn_training",
            "teacher_or_demo_data": False,
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = f"{tmpdir}/image.pth"
            save_checkpoint(
                path,
                model,
                no_duck=True,
                no_duck_margin=0.01,
                training_config=config,
                provenance=provenance,
            )
            restored, device, spec = load_image_policy(path, "cpu")
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(payload["checkpoint_format"], IMAGE_CHECKPOINT_FORMAT)
        self.assertEqual(payload["encoder"], "impala")
        self.assertEqual(payload["state_shape"], (4, 16, 32))
        self.assertEqual(payload["action_names"], IMAGE_ACTION_NAMES)
        self.assertEqual(payload["training_config"], config)
        self.assertEqual(payload["provenance"], provenance)
        self.assertEqual(spec["state_shape"], (4, 16, 32))
        self.assertTrue(spec["no_duck"])
        self.assertFalse(restored.training)

    def test_legacy_policy_spec_uses_promoted_checkpoint_defaults(self):
        checkpoint = {
            "model_state_dict": {
                "advantage.bias": torch.zeros(3),
            },
        }

        spec = image_policy_spec(checkpoint)

        self.assertEqual(spec["state_shape"], (4, 84, 336))
        self.assertEqual(spec["encoder"], "impala")
        self.assertEqual(spec["action_size"], 3)
        self.assertFalse(spec["no_duck"])

    def test_no_duck_metadata_controls_verifier_action(self):
        class FixedQ(torch.nn.Module):
            def forward(self, _obs):
                return torch.tensor([[0.0, 2.0, 100.0]])

        obs = np.zeros((4, 2, 3), dtype=np.float32)
        model = FixedQ()

        self.assertEqual(policy_action(model, obs, "cpu", no_duck=False), 2)
        self.assertEqual(policy_action(model, obs, "cpu", no_duck=True), 1)

    def test_training_metadata_records_policy_config_and_data_lineage(self):
        args = argparse.Namespace(
            encoder="nature",
            no_duck=False,
            no_duck_margin=0.01,
            deterministic=True,
            frames_per_action=1,
            init_from="base.pth",
        )
        demo = {"format": "demo-v1", "teacher_or_demo_data": True}
        self_anchor = {
            "kind": "self_generated_replay",
            "teacher_or_demo_data": False,
        }

        metadata = image_checkpoint_metadata(
            args, (4, 84, 336), demo, self_anchor
        )

        self.assertEqual(metadata["encoder"], "nature")
        self.assertEqual(metadata["action_mode"], "noop_jump_duck")
        self.assertEqual(
            metadata["training_config"]["state_shape"], (4, 84, 336)
        )
        self.assertTrue(metadata["provenance"]["teacher_or_demo_data"])
        self.assertEqual(metadata["provenance"]["demo_bank"], demo)
        self.assertEqual(
            metadata["provenance"]["self_replay_anchor"], self_anchor
        )

    def test_demo_bank_metadata_is_safe_json_with_legacy_fallback(self):
        metadata = {
            "format": "demo-v1",
            "teacher": {"sha256": "abc"},
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            modern_path = f"{tmpdir}/modern.npz"
            legacy_path = f"{tmpdir}/legacy.npz"
            np.savez(
                modern_path,
                metadata_json=np.array(json.dumps(metadata)),
            )
            np.savez(legacy_path, actions=np.zeros(1, dtype=np.int64))
            with np.load(modern_path) as archive:
                modern = demo_bank_metadata(archive, modern_path)
            with np.load(legacy_path) as archive:
                legacy = demo_bank_metadata(archive, legacy_path)

        self.assertEqual(modern["teacher"]["sha256"], "abc")
        self.assertTrue(modern["teacher_or_demo_data"])
        self.assertEqual(
            legacy["format"], "legacy_unversioned_image_demo_bank"
        )

    def test_harvester_defaults_and_teacher_provenance_are_corrected(self):
        args = build_harvest_parser().parse_args([])
        self.assertEqual(args.teacher, DEFAULT_TEACHER)
        self.assertEqual(args.out, DEFAULT_OUTPUT)

        teacher = DuelingDQN(10, 3)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = f"{tmpdir}/teacher.pth"
            torch.save(
                {
                    "model": teacher.state_dict(),
                    "no_duck": True,
                    "env_backend": "browser",
                    "observation_mode": "features",
                },
                path,
            )
            restored, action_mode, provenance = load_teacher(path, "cpu")

        self.assertFalse(restored.training)
        self.assertTrue(action_mode["no_duck"])
        self.assertEqual(len(provenance["sha256"]), 64)
        self.assertEqual(
            provenance["kind"], "corrected_browser_feature_dqn_checkpoint"
        )

    def test_bank_metadata_captures_collection_contract(self):
        args = build_harvest_parser().parse_args([])
        actions = np.array([0, 1, 1], dtype=np.int64)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        expert = np.array([True, True, False])

        metadata = build_bank_metadata(
            args, {"sha256": "abc"}, actions, dones, expert
        )

        self.assertEqual(metadata["state_shape"], (4, 84, 336))
        self.assertEqual(metadata["environment"]["frames_per_action"], 1)
        self.assertEqual(metadata["collection"]["action_counts"], [1, 2, 0])
        self.assertEqual(metadata["collection"]["expert_transitions"], 2)
        self.assertEqual(metadata["collection"]["terminal_transitions"], 1)

    def test_live_resume_request_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_resume_request(
                True,
                "missing-live.pt",
                "",
                path_exists=lambda _path: False,
            )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_resume_request(
                True,
                "existing-live.pt",
                "warm-start.pth",
                path_exists=lambda _path: True,
            )

    def test_replay_snapshot_schema_validation(self):
        capacity = 3
        state_shape = (4, 2, 3)
        snapshot = {
            "obs": torch.zeros((capacity, *state_shape), dtype=torch.uint8),
            "next_obs": torch.zeros(
                (capacity, *state_shape), dtype=torch.uint8
            ),
            "actions": torch.zeros(capacity, dtype=torch.long),
            "rewards": torch.zeros(capacity),
            "dones": torch.zeros(capacity),
            "disc": torch.full((capacity,), 0.99),
            "idx": 1,
            "size": 2,
            "online_sd": {},
            "target_sd": {},
            "opt_sd": {},
        }

        self.assertIsNone(validate_replay_snapshot(
            snapshot, capacity, state_shape, require_live=True
        ))

        missing = dict(snapshot)
        del missing["disc"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_replay_snapshot(
                missing, capacity, state_shape, require_live=True
            )

        bad_shape = dict(snapshot)
        bad_shape["next_obs"] = torch.zeros(
            (capacity, 4, 2, 4), dtype=torch.uint8
        )
        with self.assertRaisesRegex(ValueError, "next_obs shape"):
            validate_replay_snapshot(
                bad_shape, capacity, state_shape, require_live=True
            )


if __name__ == "__main__":
    unittest.main()
