import tempfile
import unittest

import numpy as np
import torch

from extract_self_image_bank import (
    build_parser,
    chronological_replay_indices,
    select_anchor_indices,
    terminal_replay_indices,
)
from image_dqn import (
    SELF_REPLAY_ANCHOR_FORMAT,
    adjusted_live_terminal_frac,
    dqfd_margin_loss,
    load_self_replay_anchor,
    replay_batch_sizes,
    validate_self_replay_anchor,
)


def make_anchor_payload(size=4):
    obs = torch.arange(size * 6, dtype=torch.uint8).reshape(size, 1, 2, 3)
    return {
        "format": SELF_REPLAY_ANCHOR_FORMAT,
        "size": size,
        "capacity": size,
        "idx": 0,
        "obs": obs,
        "next_obs": obs.flip(0),
        "actions": torch.arange(size, dtype=torch.long) % 3,
        "rewards": torch.linspace(0.0, 0.3, size),
        "dones": torch.zeros(size),
        "disc": torch.full((size,), 0.99),
        "provenance": {
            "kind": "self_generated_replay",
            "teacher_or_demo_data": False,
            "seed": 17,
        },
    }


class SelfReplayAnchorTest(unittest.TestCase):
    def test_batch_mix_preserves_defaults_and_total_terminal_quota(self):
        self.assertEqual(
            replay_batch_sizes(512, False, 0.25, False, 0.0),
            (0, 0, 512),
        )
        self.assertEqual(
            adjusted_live_terminal_frac(512, 512, 0.1),
            0.1,
        )

        mix = replay_batch_sizes(512, True, 0.25, True, 0.25)
        self.assertEqual(mix, (128, 128, 256))
        live_frac = adjusted_live_terminal_frac(512, mix[2], 0.1)
        self.assertEqual(int(mix[2] * live_frac), int(512 * 0.1))

        with self.assertRaisesRegex(ValueError, "terminal quota"):
            adjusted_live_terminal_frac(100, 20, 0.5)

    def test_self_rows_cannot_enter_demo_margin_loss(self):
        actions = torch.tensor([1, 0, 2])
        q_values = torch.tensor([
            [0.0, 0.25, 0.5],
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
        ])
        expected = dqfd_margin_loss(q_values, actions, 1, margin=0.8)

        changed_non_demo = q_values.clone()
        changed_non_demo[1:] = torch.tensor([
            [-1000.0, 1000.0, 500.0],
            [1000.0, -1000.0, 500.0],
        ])
        actual = dqfd_margin_loss(
            changed_non_demo, actions, 1, margin=0.8
        )

        self.assertEqual(actual.item(), expected.item())

    def test_anchor_validation_and_cpu_staged_load(self):
        payload = make_anchor_payload()
        size, provenance = validate_self_replay_anchor(
            payload, state_shape=(1, 2, 3)
        )
        self.assertEqual(size, 4)
        self.assertEqual(provenance["kind"], "self_generated_replay")

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = f"{tmpdir}/anchor.pt"
            torch.save(payload, path)
            replay, loaded_provenance = load_self_replay_anchor(
                path, (1, 2, 3), "cpu", chunk_size=2
            )

        self.assertEqual(replay.size, 4)
        self.assertEqual(replay.capacity, 4)
        self.assertTrue(torch.equal(replay.obs, payload["obs"]))
        self.assertTrue(torch.equal(replay.next_obs, payload["next_obs"]))
        self.assertEqual(loaded_provenance, payload["provenance"])

        payload["provenance"]["teacher_or_demo_data"] = True
        with self.assertRaisesRegex(ValueError, "teacher/demo"):
            validate_self_replay_anchor(payload)

    def test_stratified_selection_is_reproducible_across_wrapped_replay(self):
        order = chronological_replay_indices(10, 10, 4)
        np.testing.assert_array_equal(order, [4, 5, 6, 7, 8, 9, 0, 1, 2, 3])

        rewards = np.zeros(10, dtype=np.float32)
        rewards[[4, 5, 6]] = 1.0
        dones = np.zeros(10, dtype=np.float32)
        kwargs = {
            "size": 10,
            "write_index": 4,
            "target_size": 6,
            "recent_frac": 0.5,
            "recent_window": 4,
            "clear_reward_min": 0.5,
            "seed": 731,
        }
        selected_a, stats_a = select_anchor_indices(
            rewards, dones, **kwargs
        )
        selected_b, stats_b = select_anchor_indices(
            rewards, dones, **kwargs
        )

        np.testing.assert_array_equal(selected_a, selected_b)
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(len(np.unique(selected_a)), 6)
        self.assertTrue(set([4, 5, 6]).issubset(selected_a))
        self.assertEqual(stats_a["assigned_clear_rows"], 3)
        self.assertEqual(stats_a["assigned_recent_rows"], 3)
        self.assertEqual(stats_a["selected_recent_rows"], 3)

    def test_terminal_pool_uses_only_done_rows_in_interleaved_order(self):
        dones = np.zeros(12, dtype=np.float32)
        dones[[1, 5, 9, 11]] = 1.0

        pool = terminal_replay_indices(dones, size=10, write_index=10)

        # Non-terminal rows between each env's n-step flush rows never enter.
        # Physical row 11 is also excluded because it has not been written.
        np.testing.assert_array_equal(pool, [1, 5, 9])
        self.assertTrue(np.all(dones[pool] >= 0.5))

    def test_terminal_quota_underfills_then_general_fill_completes_anchor(self):
        rewards = np.linspace(0.0, 0.9, 10, dtype=np.float32)
        dones = np.zeros(10, dtype=np.float32)
        dones[[5, 9]] = 1.0
        kwargs = {
            "size": 10,
            "write_index": 0,
            "target_size": 6,
            "recent_frac": 0.5,
            "recent_window": 4,
            "clear_reward_min": 0.5,
            "seed": 91,
        }

        legacy_selected, legacy_stats = select_anchor_indices(
            rewards, dones, **kwargs
        )
        zero_selected, zero_stats = select_anchor_indices(
            rewards, dones, terminal_frac=0.0, **kwargs
        )
        np.testing.assert_array_equal(legacy_selected, zero_selected)
        self.assertEqual(legacy_stats, zero_stats)
        self.assertNotIn("terminal_requested_rows", legacy_stats)

        selected, stats = select_anchor_indices(
            rewards, dones, terminal_count=4, **kwargs
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(np.unique(selected)), 6)
        self.assertEqual(stats["terminal_requested_rows"], 4)
        self.assertEqual(stats["terminal_assigned_rows"], 2)
        self.assertEqual(stats["terminal_pool_size"], 2)
        self.assertEqual(stats["selected_terminal_rows"], 2)
        self.assertEqual(np.count_nonzero(dones[selected] >= 0.5), 2)

        frac_selected, frac_stats = select_anchor_indices(
            rewards, dones, terminal_frac=2 / 3, **kwargs
        )
        np.testing.assert_array_equal(frac_selected, selected)
        self.assertEqual(frac_stats, stats)

    def test_extractor_defaults_document_a_30k_balanced_anchor(self):
        args = build_parser().parse_args(["source.pt", "anchor.pt"])
        self.assertEqual(args.size, 30000)
        self.assertEqual(args.recent_frac, 0.5)
        self.assertEqual(args.recent_window, 60000)
        self.assertEqual(args.clear_reward_min, 0.15)
        self.assertEqual(args.terminal_frac, 0.0)
        self.assertIsNone(args.terminal_count)


if __name__ == "__main__":
    unittest.main()
