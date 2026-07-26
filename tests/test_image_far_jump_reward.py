from collections import deque
import unittest

import numpy as np

from dino_rl.browser_env import ChromeDinoImageEnv
from dino_rl.feature_contract import FEATURE_DIM, GAME_WIDTH
from image_dqn import (
    apply_far_jump_penalty,
    build_parser,
    far_jump_penalty_mask,
    far_jump_reward_metadata,
)


def browser_features(distance_pixels=200.0):
    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    features[0] = distance_pixels / GAME_WIDTH
    features[1] = 0.2
    features[2] = 0.4
    return features


class ImageFarJumpRewardTest(unittest.TestCase):
    def test_image_env_info_copies_features_before_deterministic_action(self):
        pre_features = browser_features(300.0).tolist()
        pre_state = {
            "features": pre_features,
            "score": 17,
            "distanceRan": 10.0,
        }
        post_state = {
            "features": browser_features(180.0).tolist(),
            "score": 18,
            "distanceRan": 12.0,
            "crashed": False,
        }

        class FakeGame:
            def env_step(self, action, frames):
                self.action = action
                self.frames = frames
                pre_features[0] = 0.01
                return post_state

        env = ChromeDinoImageEnv.__new__(ChromeDinoImageEnv)
        env.game = FakeGame()
        env.frames_per_action = 2
        env.frame_stack = 4
        env.obs_h = 2
        env.obs_w = 3
        env._frames = deque(
            [np.zeros((2, 3), dtype=np.float32) for _ in range(4)],
            maxlen=4,
        )
        env._last_state = pre_state
        env._last_distance_ran = 10.0
        env.reward_mode = "survival"
        env.survival_reward = 0.01
        env.distance_reward_scale = 0.02
        env.gameover_penalty = -1.0
        env._capture_frame = lambda: np.ones((2, 3), dtype=np.float32)

        obs, _reward, done, info = env._step_deterministic(1)

        self.assertFalse(done)
        self.assertEqual(obs.shape, (4, 2, 3))
        self.assertEqual(info["pre_action_features"][0], 0.5)
        self.assertEqual(env.game.action, 1)
        self.assertEqual(env.game.frames, 2)

    def test_mask_requires_exact_grounded_visible_far_jump_state(self):
        eligible = browser_features(200.0)
        rows = [eligible.copy() for _ in range(10)]
        actions = np.ones(len(rows), dtype=np.int64)
        actions[1] = 0
        rows[2][0] = 199.9 / GAME_WIDTH
        rows[3][1] = 0.0
        rows[4][2] = 0.0
        rows[5][3] = 1e-12
        rows[6][4] = -1e-12
        rows[7][5] = 1.0
        rows[8][9] = 1.0
        rows[9][0] = np.nan
        infos = [{"pre_action_features": row} for row in rows]

        mask = far_jump_penalty_mask(actions, infos, 200.0)

        np.testing.assert_array_equal(
            mask,
            [True, False, False, False, False, False, False, False, False, False],
        )

    def test_penalty_is_opt_in_and_subtracted_only_from_masked_rows(self):
        actions = np.array([1, 1, 0], dtype=np.int64)
        infos = [
            {"pre_action_features": browser_features(250.0)},
            {"pre_action_features": browser_features(150.0)},
            {"pre_action_features": browser_features(250.0)},
        ]
        rewards = np.array([0.01, 0.01, -10.0], dtype=np.float32)

        disabled = apply_far_jump_penalty(rewards, actions, infos, 0.0, 200.0)
        adjusted = apply_far_jump_penalty(rewards, actions, infos, 0.5, 200.0)

        self.assertIs(disabled, rewards)
        np.testing.assert_array_equal(disabled, rewards)
        np.testing.assert_allclose(adjusted, [-0.49, 0.01, -10.0])

    def test_cli_defaults_and_metadata_keep_features_reward_only(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.far_jump_penalty, 0.0)
        self.assertEqual(args.far_jump_distance, 200.0)

        metadata = far_jump_reward_metadata(0.25, 200.0)
        self.assertEqual(metadata["far_jump_penalty"], 0.25)
        self.assertEqual(metadata["far_jump_distance"], 200.0)
        self.assertEqual(
            metadata["far_jump_reward_provenance"]["policy_observation"],
            "pixels_only",
        )

    def test_config_rejects_negative_or_nonfinite_values(self):
        with self.assertRaises(ValueError):
            far_jump_reward_metadata(-0.1, 200.0)
        with self.assertRaises(ValueError):
            far_jump_reward_metadata(0.1, np.inf)


if __name__ == "__main__":
    unittest.main()
