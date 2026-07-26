import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from unittest.mock import patch

import numpy as np
import torch

import dino_rl.train_dqn as train_dqn
from dino_rl.networks import DuelingDQN
from train_browser_dqn import (
    NStepAccumulator,
    VecFeatureBrowser,
    _maybe_apply_speed_curriculum,
    clone_feature_model_to_cpu,
    evaluate,
    sample_random_feature_actions,
)


def state(value=0.0):
    return np.full(10, value, dtype=np.float32)


def sequence_state(
    dist1_pixels=175.0,
    dist2_pixels=550.0,
    speed=12.5,
):
    features = state()
    features[0] = dist1_pixels / train_dqn.GAME_WIDTH
    features[1] = 17.0 / train_dqn.MAX_OBSTACLE_WIDTH
    features[2] = 35.0 / train_dqn.MAX_OBSTACLE_HEIGHT
    features[6] = dist2_pixels / train_dqn.GAME_WIDTH
    features[7] = speed / train_dqn.MAX_GAME_SPEED
    features[8] = 105.0 / train_dqn.GAME_HEIGHT
    return features


class FeatureReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_device = train_dqn.device
        train_dqn.device = torch.device("cpu")

    @classmethod
    def tearDownClass(cls):
        train_dqn.device = cls.original_device

    def test_terminal_cache_tracks_eviction_and_reserves_quota(self):
        agent = train_dqn.Agent(3, memory_capacity=4)
        agent.remember(state(0), 0, -1.0, state(1), True)
        agent.remember(state(1), 0, 0.0, state(2), False)
        agent.remember(state(2), 1, 0.0, state(3), False)
        agent.remember(state(3), 1, -1.0, state(4), True)

        batch = agent.sample_replay(4, terminal_frac=0.5)

        self.assertEqual(len(batch), 4)
        self.assertGreaterEqual(sum(bool(row[4]) for row in batch), 2)
        self.assertEqual(len(agent.terminal_memory), 2)

        agent.remember(state(4), 0, 0.0, state(5), False)
        self.assertEqual(len(agent.terminal_memory), 1)
        self.assertIs(agent.terminal_memory[0], agent.memory[2])

    def test_sequence_cache_tracks_eviction_and_reserves_quota(self):
        agent = train_dqn.Agent(3, memory_capacity=6)
        agent.remember(sequence_state(speed=12.6), 0, 0.0, state(1), False)
        agent.remember(
            sequence_state(dist1_pixels=180.0, speed=12.6),
            1, 0.0, state(2), False
        )
        for i in range(4):
            agent.remember(state(i), 0, 0.0, state(i + 1), False)

        batch = agent.sample_replay(4, sequence_frac=0.5)

        self.assertEqual(len(batch), 4)
        self.assertGreaterEqual(
            sum(train_dqn.is_forensic_sequence_jump_state(row[0]) for row in batch),
            2,
        )
        self.assertEqual(len(agent.sequence_memory), 2)
        agent.remember(state(9), 0, 0.0, state(10), False)
        self.assertEqual(len(agent.sequence_memory), 1)

    def test_reserved_replay_sampling_always_returns_a_full_batch(self):
        agent = train_dqn.Agent(3, memory_capacity=40)
        for i in range(20):
            agent.remember(
                sequence_state(dist1_pixels=175.0 + i),
                1,
                -1.0 if i % 2 else 0.0,
                state(i + 1),
                bool(i % 2),
            )
        for i in range(20, 40):
            agent.remember(state(i), 0, 0.0, state(i + 1), False)

        for _ in range(100):
            batch = agent.sample_replay(
                16, terminal_frac=0.25, sequence_frac=0.25
            )
            self.assertEqual(len(batch), 16)
            self.assertEqual(len({id(row) for row in batch}), 16)

    def test_replay_accepts_legacy_and_discounted_rows(self):
        agent = train_dqn.Agent(3, memory_capacity=4)
        agent.remember(state(0), 0, 0.1, state(1), False)
        agent.remember(state(1), 1, 0.2, state(2), False, 0.81)
        agent.remember(state(2), 0, -1.0, state(3), True)
        agent.remember(state(3), 1, 0.3, state(4), False, 0.729)

        self.assertEqual([len(row) for row in agent.memory], [5, 6, 5, 6])
        agent.replay(4, terminal_frac=0.25)

    def test_grounded_empty_state_mask_uses_feature_contract_semantics(self):
        states = torch.zeros(5, 10)
        states[:, 0] = 1.0
        states[1, 0] = 0.99995
        states[1, 1] = 0.00005
        states[1, 2] = -0.00005
        states[2, 5] = 1.0
        states[3, 0] = 0.4
        states[3, 1:3] = 0.2
        states[4, 1] = 0.01

        mask = train_dqn.grounded_empty_state_mask(states)

        self.assertTrue(torch.equal(
            mask, torch.tensor([True, True, False, False, False])
        ))

    def test_empty_state_noop_hinge_ignores_nonempty_and_airborne_states(self):
        states = torch.zeros(4, 10)
        states[:, 0] = 1.0
        states[2, 0] = 0.4
        states[2, 1:3] = 0.2
        states[3, 5] = 1.0
        q_values = torch.tensor(
            [
                [0.0, 0.02, 99.0],
                [0.2, 0.1, 99.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ],
            requires_grad=True,
        )

        loss = train_dqn.empty_state_noop_hinge_loss(
            q_values, states, margin=0.05
        )

        self.assertAlmostEqual(loss.item(), 0.035, places=6)
        loss.backward()
        expected = torch.tensor(
            [
                [-0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(q_values.grad, expected))


    def test_grounded_visible_far_mask_uses_real_replay_semantics(self):
        states = torch.zeros(9, 10)
        states[:, 0] = 300.0 / train_dqn.GAME_WIDTH
        states[:, 1] = 0.2
        states[:, 2] = 0.4
        states[1, 3] = 0.00005
        states[1, 4] = -0.00005
        states[2, 0] = 299.0 / train_dqn.GAME_WIDTH
        states[3, 3] = 0.001
        states[4, 4] = -0.001
        states[5, 5] = 1.0
        states[6, 9] = 1.0
        states[7, 1] = 0.0
        states[8, 2] = 0.0

        mask = train_dqn.grounded_visible_far_state_mask(
            states, distance_pixels=300.0
        )

        self.assertTrue(torch.equal(
            mask,
            torch.tensor([
                True, True, False, False, False, False, False, False, False
            ]),
        ))

    def test_far_state_noop_hinge_ignores_near_and_moving_states(self):
        states = torch.zeros(4, 10)
        states[:, 0] = 400.0 / train_dqn.GAME_WIDTH
        states[:, 1] = 0.2
        states[:, 2] = 0.4
        states[2, 0] = 200.0 / train_dqn.GAME_WIDTH
        states[3, 4] = 0.1
        q_values = torch.tensor(
            [
                [0.0, 0.02, 99.0],
                [0.2, 0.1, 99.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ],
            requires_grad=True,
        )

        loss = train_dqn.far_state_noop_hinge_loss(
            q_values, states, distance_pixels=300.0, margin=0.05
        )

        self.assertAlmostEqual(loss.item(), 0.035, places=6)
        loss.backward()
        expected = torch.tensor(
            [
                [-0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(q_values.grad, expected))

    def test_forensic_sequence_mask_has_inclusive_distance_boundaries(self):
        rows = [
            sequence_state(150.0, 525.0, 12.5),
            sequence_state(200.0, 590.0, 13.0),
            sequence_state(149.9, 550.0, 13.0),
            sequence_state(200.1, 550.0, 13.0),
            sequence_state(175.0, 524.9, 13.0),
            sequence_state(175.0, 590.1, 13.0),
            sequence_state(175.0, 550.0, 12.49),
        ]
        wrong_width = sequence_state()
        wrong_width[1] += 2e-4
        rows.append(wrong_width)
        wrong_height = sequence_state()
        wrong_height[2] += 2e-4
        rows.append(wrong_height)
        wrong_y = sequence_state()
        wrong_y[8] += 2e-4
        rows.append(wrong_y)
        for index in (3, 4, 5, 9):
            not_grounded = sequence_state()
            not_grounded[index] = 0.001 if index in (3, 4) else 1.0
            rows.append(not_grounded)

        mask = train_dqn.forensic_sequence_jump_state_mask(
            torch.as_tensor(np.stack(rows))
        )

        self.assertTrue(torch.equal(
            mask,
            torch.tensor([True, True] + [False] * (len(rows) - 2)),
        ))

    def test_sequence_jump_hinge_requires_jump_over_noop(self):
        states = torch.as_tensor(np.stack([
            sequence_state(),
            sequence_state(200.0, 590.0, 13.0),
            sequence_state(175.0, 500.0, 13.0),
        ]))
        q_values = torch.tensor(
            [[0.1, 0.0, 99.0], [0.0, 0.2, 99.0], [100.0, 0.0, 0.0]],
            requires_grad=True,
        )

        loss = train_dqn.sequence_jump_hinge_loss(
            q_values, states, margin=0.05
        )

        self.assertAlmostEqual(loss.item(), 0.075, places=6)
        loss.backward()
        expected = torch.tensor(
            [[0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        self.assertTrue(torch.allclose(q_values.grad, expected))


    def test_default_replay_does_not_compute_regularizer_losses(self):
        agent = train_dqn.Agent(3, memory_capacity=4)
        for i in range(4):
            agent.remember(
                state(i), i % 3, float(i), state(i + 1), i == 3
            )

        with patch.object(
            train_dqn,
            "empty_state_noop_hinge_loss",
            side_effect=AssertionError("disabled empty-state loss was evaluated"),
        ) as empty_noop_loss, patch.object(
            train_dqn,
            "far_state_noop_hinge_loss",
            side_effect=AssertionError("disabled far-state loss was evaluated"),
        ) as far_noop_loss, patch.object(
            train_dqn,
            "sequence_jump_hinge_loss",
            side_effect=AssertionError("disabled sequence loss was evaluated"),
        ) as sequence_loss:
            agent.replay(4)

        self.assertEqual(agent.far_state_noop_distance, 0.0)
        self.assertEqual(agent.far_state_noop_margin, 0.0)
        self.assertEqual(agent.far_state_noop_weight, 0.0)
        self.assertEqual(agent.sequence_jump_margin, 0.0)
        self.assertEqual(agent.sequence_jump_weight, 0.0)
        empty_noop_loss.assert_not_called()
        far_noop_loss.assert_not_called()
        sequence_loss.assert_not_called()

    def test_enabled_replay_applies_empty_state_noop_hinge(self):
        agent = train_dqn.Agent(
            3,
            memory_capacity=4,
            empty_state_noop_margin=0.05,
            empty_state_noop_weight=0.1,
        )
        empty = state()
        empty[0] = 1.0
        for i in range(4):
            agent.remember(
                empty.copy(), i % 2, float(i), empty.copy(), i == 3
            )

        with patch.object(
            train_dqn,
            "empty_state_noop_hinge_loss",
            wraps=train_dqn.empty_state_noop_hinge_loss,
        ) as empty_noop_loss:
            agent.replay(4)

        empty_noop_loss.assert_called_once()
        self.assertEqual(empty_noop_loss.call_args.args[2], 0.05)


    def test_enabled_replay_applies_far_state_noop_hinge(self):
        agent = train_dqn.Agent(
            3,
            memory_capacity=4,
            far_state_noop_distance=300.0,
            far_state_noop_margin=0.05,
            far_state_noop_weight=0.1,
        )
        far_state = state()
        far_state[0] = 400.0 / train_dqn.GAME_WIDTH
        far_state[1] = 0.2
        far_state[2] = 0.4
        for i in range(4):
            agent.remember(
                far_state.copy(),
                i % 2,
                float(i),
                far_state.copy(),
                i == 3,
            )

        with patch.object(
            train_dqn,
            "far_state_noop_hinge_loss",
            wraps=train_dqn.far_state_noop_hinge_loss,
        ) as far_noop_loss:
            agent.replay(4)

        far_noop_loss.assert_called_once()
        self.assertEqual(far_noop_loss.call_args.args[2:], (300.0, 0.05))

    def test_sequence_hinge_requires_positive_margin_and_weight(self):
        agents = [
            train_dqn.Agent(
                3, memory_capacity=4,
                sequence_jump_margin=0.05, sequence_jump_weight=0.0,
            ),
            train_dqn.Agent(
                3, memory_capacity=4,
                sequence_jump_margin=0.0, sequence_jump_weight=0.1,
            ),
        ]
        for agent in agents:
            for i in range(4):
                agent.remember(
                    sequence_state(), i % 2, float(i), sequence_state(), i == 3
                )

        with patch.object(
            train_dqn,
            "sequence_jump_hinge_loss",
            side_effect=AssertionError("partially disabled loss was evaluated"),
        ) as sequence_loss:
            for agent in agents:
                agent.replay(4)

        sequence_loss.assert_not_called()

    def test_enabled_replay_applies_sequence_jump_hinge(self):
        agent = train_dqn.Agent(
            3,
            memory_capacity=4,
            sequence_jump_margin=0.05,
            sequence_jump_weight=0.1,
        )
        for i in range(4):
            agent.remember(
                sequence_state(), i % 2, float(i), sequence_state(), i == 3
            )

        with patch.object(
            train_dqn,
            "sequence_jump_hinge_loss",
            wraps=train_dqn.sequence_jump_hinge_loss,
        ) as sequence_loss:
            agent.replay(4)

        sequence_loss.assert_called_once()
        self.assertEqual(sequence_loss.call_args.args[2], 0.05)

    def test_sequence_configuration_rejects_nonfinite_or_negative_values(self):
        with self.assertRaisesRegex(ValueError, "sequence_jump_margin"):
            train_dqn.Agent(3, sequence_jump_margin=-0.01)
        with self.assertRaisesRegex(ValueError, "sequence_jump_weight"):
            train_dqn.Agent(3, sequence_jump_weight=np.inf)


    def test_regularizer_configuration_is_saved(self):
        agent = train_dqn.Agent(
            3,
            empty_state_noop_margin=0.05,
            empty_state_noop_weight=0.1,
            far_state_noop_distance=300.0,
            far_state_noop_margin=0.05,
            far_state_noop_weight=0.1,
            sequence_jump_margin=0.02,
            sequence_jump_weight=0.25,
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = f"{tmpdir}/gap.pth"
            agent.save_model(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["empty_state_noop_margin"], 0.05)
        self.assertEqual(checkpoint["empty_state_noop_weight"], 0.1)
        self.assertEqual(checkpoint["far_state_noop_distance"], 300.0)
        self.assertEqual(checkpoint["far_state_noop_margin"], 0.05)
        self.assertEqual(checkpoint["far_state_noop_weight"], 0.1)
        self.assertEqual(checkpoint["sequence_jump_margin"], 0.02)
        self.assertEqual(checkpoint["sequence_jump_weight"], 0.25)


    def test_no_duck_projection_survives_learning_and_raw_save(self):
        margin = 0.01
        agent = train_dqn.Agent(
            3, memory_capacity=4, no_duck=True, no_duck_margin=margin
        )
        agent.epsilon = 1.0

        for _ in range(50):
            self.assertNotEqual(agent.act(state()), 2)
        self.assertFalse(agent.remember(state(), 2, 0.0, state(1), False))

        for i in range(4):
            agent.remember(
                state(i), i % 2, -1.0 if i == 3 else 0.01,
                state(i + 1), i == 3,
            )
        agent.replay(4, terminal_frac=0.25)

        for model in (agent.model, agent.target_model):
            self.assertTrue(torch.equal(
                model.advantage.weight[2], model.advantage.weight[0]
            ))
            self.assertTrue(torch.allclose(
                model.advantage.bias[2],
                model.advantage.bias[0] - margin,
            ))

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = f"{tmpdir}/projected.pth"
            with torch.no_grad():
                agent.model.advantage.bias[2].add_(100.0)
            agent.save_model(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        raw_model = DuelingDQN(10, 3)
        raw_model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            q_values = raw_model(torch.randn(64, 10))
        self.assertFalse(bool((q_values.argmax(dim=1) == 2).any()))
        self.assertTrue(checkpoint["no_duck"])


class NStepAccumulatorTest(unittest.TestCase):
    def test_emits_discounted_window_and_flushes_terminal_suffixes(self):
        accumulator = NStepAccumulator(3, 0.9)
        self.assertEqual(accumulator.push("s0", 0, 1.0, "s1", False), [])
        self.assertEqual(accumulator.push("s1", 1, 2.0, "s2", False), [])
        rows = accumulator.push("s2", 0, 3.0, "s3", False)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][2], 1.0 + 0.9 * 2.0 + 0.81 * 3.0)
        self.assertEqual(rows[0][3], "s3")
        self.assertFalse(rows[0][4])
        self.assertAlmostEqual(rows[0][5], 0.9 ** 3)

        terminal = NStepAccumulator(3, 0.9)
        terminal.push("s0", 0, 1.0, "s1", False)
        rows = terminal.push("s1", 1, -10.0, "dead", True)

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0][2], -8.0)
        self.assertAlmostEqual(rows[0][5], 0.9 ** 2)
        self.assertTrue(rows[0][4])
        self.assertAlmostEqual(rows[1][2], -10.0)
        self.assertAlmostEqual(rows[1][5], 0.9)
        self.assertEqual(len(terminal.buffer), 0)

    def test_clear_discards_interrupted_window(self):
        accumulator = NStepAccumulator(3, 0.99)
        accumulator.push("old", 0, 0.01, "old-next", False)
        accumulator.clear()

        self.assertEqual(accumulator.push("new", 0, 0.01, "n1", False), [])
        self.assertEqual(len(accumulator.buffer), 1)


class CPUEvaluationTest(unittest.TestCase):
    def test_cpu_clone_is_independent_and_preserves_live_model_state(self):
        live_model = DuelingDQN(10, 3)
        live_model.train()
        inputs = torch.randn(4, 10)
        live_parameter = next(live_model.parameters())
        parameter_before = live_parameter.detach().clone()
        with torch.no_grad():
            expected = live_model(inputs).clone()

        cpu_model = clone_feature_model_to_cpu(live_model)

        self.assertTrue(live_model.training)
        self.assertFalse(cpu_model.training)
        self.assertTrue(all(p.device.type == "cpu" for p in cpu_model.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.equal(cpu_model(inputs), expected))
        cpu_parameter = next(cpu_model.parameters())
        self.assertNotEqual(cpu_parameter.data_ptr(), live_parameter.data_ptr())
        with torch.no_grad():
            cpu_parameter.add_(1.0)
        self.assertTrue(torch.equal(live_parameter, parameter_before))

    def test_browser_evaluation_parallelizes_cpu_snapshot_workers(self):
        class CPUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_lock = Lock()

            def forward(self, inputs):
                if inputs.device.type != "cpu":
                    raise AssertionError("evaluation input was not on CPU")
                if not self.forward_lock.acquire(blocking=False):
                    raise AssertionError("shared CPU inference overlapped")
                try:
                    sleep(0.01)
                    return torch.tensor(
                        [[0.0, 1.0, 99.0]], device=inputs.device
                    ).repeat(inputs.shape[0], 1)
                finally:
                    self.forward_lock.release()

        class FakeGame:
            instances = []
            start_barrier = Barrier(2)

            def __init__(self, headless=True):
                self.calls = 0
                self.actions = []
                self.closed = False
                type(self).instances.append(self)

            def start(self):
                type(self).start_barrier.wait(timeout=2)

            def enable_deterministic(self):
                pass

            def restart_deterministic(self):
                self.calls = 0

            def env_step(self, action, _frames):
                self.actions.append(action)
                self.calls += 1
                if self.calls == 1:
                    return {
                        "features": state(), "crashed": False, "score": 0
                    }
                return {
                    "features": state(), "crashed": True, "score": 123
                }

            def close(self):
                self.closed = True

        class FakeAgent:
            model = object()
            no_duck = True

            def act(self, *_args, **_kwargs):
                raise AssertionError("live agent inference was used")

        FakeGame.instances = []
        FakeGame.start_barrier = Barrier(2)
        agent = FakeAgent()
        with patch(
            "train_browser_dqn.clone_feature_model_to_cpu",
            return_value=CPUModel(),
        ) as clone_model, patch(
            "train_browser_dqn.ChromeDinoGame", FakeGame
        ):
            scores = evaluate(agent, episodes=2, cap=10)

        self.assertEqual(scores, [123, 123])
        self.assertEqual(len(FakeGame.instances), 2)
        for game in FakeGame.instances:
            self.assertEqual(game.actions, [0, 1])
            self.assertTrue(game.closed)
        clone_model.assert_called_once_with(agent.model)



class BrowserCollectionHelpersTest(unittest.TestCase):
    def test_random_actions_respect_no_duck_mode(self):
        features = np.zeros((200, 10), dtype=np.float32)

        actions = sample_random_feature_actions(
            features,
            no_duck=True,
            rng=np.random.RandomState(20260714),
        )

        self.assertTrue(np.all(actions < 2))
        self.assertIn(0, actions)
        self.assertIn(1, actions)

    def test_speed_curriculum_applies_only_when_selected(self):
        class Driver:
            def __init__(self):
                self.calls = []

            def execute_script(self, *args):
                self.calls.append(args)

        game = type("Game", (), {"driver": Driver()})()
        _maybe_apply_speed_curriculum(game, 0.0, 8.0, 13.0)
        self.assertEqual(game.driver.calls, [])

        with patch("train_browser_dqn.np.random.rand", return_value=0.0), \
                patch("train_browser_dqn.np.random.uniform", return_value=11.0):
            _maybe_apply_speed_curriculum(game, 1.0, 8.0, 13.0)
        self.assertEqual(game.driver.calls[-1][1], 11.0)

    def test_browser_recovery_is_an_interruption_not_a_terminal(self):
        reset_state = {"features": state(1), "crashed": False, "score": 0}
