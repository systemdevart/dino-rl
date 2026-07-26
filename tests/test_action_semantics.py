import tempfile
import unittest

import numpy as np
import torch

from dino_rl.env import DinoRunEnv
from image_dqn import (
    NStepAccumulator,
    ReplayBuffer,
    build_explorer_mask,
    eval_rank,
    greedy_action_indices,
    project_no_duck_advantage,
    save_checkpoint,
    select_epsilon_greedy_actions,
)


class ActionSemanticsTest(unittest.TestCase):
    def test_image_nstep_interruption_discards_only_affected_window(self):
        accumulator = NStepAccumulator(num_envs=2, n=2, gamma=0.9)
        first_obs = np.array([[10.0], [20.0]], dtype=np.float32)
        first_next = np.array([[11.0], [21.0]], dtype=np.float32)
        self.assertIsNone(accumulator.push(
            first_obs,
            np.array([0, 0]),
            np.array([1.0, 1.0]),
            first_next,
            np.array([False, False]),
        ))

        recovery_next = np.array([[999.0], [22.0]], dtype=np.float32)
        ready = accumulator.push(
            first_next,
            np.array([1, 1]),
            np.array([-10.0, 2.0]),
            recovery_next,
            np.array([True, False]),
            interruptions=np.array([True, False]),
        )

        self.assertIsNotNone(ready)
        obs, actions, rewards, next_obs, dones, discounts = ready
        self.assertEqual(len(actions), 1)
        np.testing.assert_array_equal(obs, [[20.0]])
        np.testing.assert_array_equal(actions, [0])
        np.testing.assert_allclose(rewards, [1.0 + 0.9 * 2.0])
        np.testing.assert_array_equal(next_obs, [[22.0]])
        np.testing.assert_array_equal(dones, [0.0])
        np.testing.assert_allclose(discounts, [0.9 ** 2])
        self.assertEqual(len(accumulator.bufs[0]), 0)

    def test_midair_duck_resets_velocity_before_speed_drop(self):
        env = DinoRunEnv(skip_clear_time=True)
        env.jumping = True
        env.dino_y = 60
        env.jump_velocity = -8.0

        env.step(2)

        self.assertTrue(env.speed_drop)
        self.assertEqual(env.dino_y, 63)
        self.assertAlmostEqual(env.jump_velocity, 1.6)
    def test_terminal_stratified_sampling_reserves_requested_fraction(self):
        replay = ReplayBuffer(10, (1, 1, 1))
        replay.size = 10
        replay.dones[9] = 1.0

        *_prefix, dones, _discounts = replay.sample(
            10, "cpu", terminal_frac=0.5
        )

        self.assertGreaterEqual(int(dones.sum().item()), 5)

    def test_action_stratified_sampling_reserves_requested_fraction(self):
        replay = ReplayBuffer(10, (1, 1, 1))
        replay.size = 10
        replay.actions[9] = 1

        _obs, actions, *_suffix = replay.sample(
            10, "cpu", action=1, action_frac=0.5
        )

        self.assertGreaterEqual(
            int((actions == 1).sum().item()),
            5,
        )


    def test_eval_rank_prioritizes_minimum_over_average(self):
        brittle = {"min": 100, "avg": 5000.0}
        consistent = {"min": 3000, "avg": 3500.0}

        self.assertGreater(eval_rank(consistent), eval_rank(brittle))


    def test_default_exploration_preserves_uniform_three_action_rng_path(self):
        greedy = np.array([2, 1, 0, 2, 1, 0], dtype=np.int64)
        np.random.seed(731)
        random_actions = np.random.randint(0, 3, size=len(greedy))
        explore = np.random.rand(len(greedy)) < 0.4
        expected = np.where(explore, random_actions, greedy)

        np.random.seed(731)
        actual = select_epsilon_greedy_actions(
            greedy, 0.4, build_explorer_mask(len(greedy), 1.0)
        )

        np.testing.assert_array_equal(actual, expected)

    def test_only_fixed_explorer_cohort_takes_biased_random_actions(self):
        greedy = np.full(4, 2, dtype=np.int64)
        actions = select_epsilon_greedy_actions(
            greedy,
            epsilon=1.0,
            explorer_mask=build_explorer_mask(4, 0.5),
            random_jump_prob=1.0,
            rng=np.random.RandomState(9),
        )

        np.testing.assert_array_equal(actions, [1, 1, 2, 2])

    def test_biased_random_actions_are_limited_to_noop_and_jump(self):
        actions = select_epsilon_greedy_actions(
            np.full(1000, 2, dtype=np.int64),
            epsilon=1.0,
            explorer_mask=np.ones(1000, dtype=bool),
            random_jump_prob=0.25,
            rng=np.random.RandomState(17),
        )

        self.assertEqual(set(np.unique(actions)), {0, 1})

    def test_no_duck_constrains_greedy_and_heterogeneous_exploration(self):
        q_values = torch.tensor([
            [0.0, 1.0, 100.0],
            [2.0, 1.0, 100.0],
        ])
        self.assertEqual(
            greedy_action_indices(q_values).tolist(),
            [2, 2],
        )
        self.assertEqual(
            greedy_action_indices(q_values, no_duck=True).tolist(),
            [1, 0],
        )

        actions = select_epsilon_greedy_actions(
            np.full(1000, 2, dtype=np.int64),
            epsilon=1.0,
            explorer_mask=build_explorer_mask(1000, 0.5),
            rng=np.random.RandomState(37),
            no_duck=True,
        )

        self.assertEqual(set(np.unique(actions[:500])), {0, 1})
        np.testing.assert_array_equal(actions[500:], np.zeros(500))

    def test_no_duck_projection_survives_optimizer_and_target_updates(self):
        class TinyDuelingQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.advantage = torch.nn.Linear(4, 3)

            def forward(self, features):
                advantage = self.advantage(features)
                return advantage - advantage.mean(dim=-1, keepdim=True)

        margin = 0.125
        online = TinyDuelingQ()
        target = TinyDuelingQ()
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=0.01)
        features = torch.randn(32, 4)

        project_no_duck_advantage(online, margin)
        project_no_duck_advantage(target, margin)
        optimizer.zero_grad()
        online(features).square().mean().backward()
        optimizer.step()
        for target_param, online_param in zip(
                target.parameters(), online.parameters()):
            target_param.data.mul_(0.9).add_(0.1 * online_param.data)
        project_no_duck_advantage(online, margin)
        project_no_duck_advantage(target, margin)

        for model in (online, target):
            self.assertTrue(torch.equal(
                model.advantage.weight[2],
                model.advantage.weight[0],
            ))
            self.assertTrue(torch.allclose(
                model.advantage.bias[2],
                model.advantage.bias[0] - margin,
            ))
            with torch.no_grad():
                q_values = model(features)
            self.assertTrue(torch.allclose(
                q_values[:, 2],
                q_values[:, 0] - margin,
                rtol=1e-6,
                atol=1e-6,
            ))
            self.assertFalse(bool((q_values.argmax(dim=-1) == 2).any()))

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            checkpoint_path = f"{tmpdir}/projected.pth"
            save_checkpoint(
                checkpoint_path,
                online,
                no_duck=True,
                no_duck_margin=margin,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

        restored = TinyDuelingQ()
        restored.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            raw_actions = restored(features).argmax(dim=-1)
        self.assertFalse(bool((raw_actions == 2).any()))
        self.assertTrue(checkpoint["no_duck"])
        self.assertEqual(checkpoint["no_duck_margin"], margin)

    def test_exploration_configuration_validates_bounds(self):
        with self.assertRaises(ValueError):
            build_explorer_mask(4, -0.01)
        with self.assertRaises(ValueError):
            build_explorer_mask(4, 1.01)
        with self.assertRaises(ValueError):
            select_epsilon_greedy_actions(
                np.zeros(4), 0.1, np.ones(4, dtype=bool),
                random_jump_prob=1.01,
            )
        with self.assertRaises(ValueError):
            project_no_duck_advantage(
                type("Model", (), {})(),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
