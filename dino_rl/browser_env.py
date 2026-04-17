"""
ChromeDriver-backed Dino environment adapted from gym-chrome-dino.

The upstream project provides the basic Selenium/ChromeDriver control flow.
This adaptation keeps that idea but uses this repo's existing feature
contract, score handling, and Chrome 145+ Runner access pattern.
"""

from __future__ import annotations

import base64
import time
from collections import deque

import cv2
import numpy as np

from selenium import webdriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from dino_rl.env import ActionSpace
from dino_rl.feature_contract import build_browser_state_js

GET_CANVAS_JS = """
Runner.instance_ = Runner.getInstance();
var canvas = document.getElementsByClassName('runner-canvas')[0];
return canvas ? canvas.toDataURL('image/png').substring(22) : null;
"""


class ChromeDinoGame:
    """Thin controller around a real Chrome Dino instance."""

    def __init__(
        self,
        *,
        page_url: str = "chrome://dino",
        headless: bool = True,
        accelerate: bool = False,
        window_size: str = "800,600",
    ):
        options = Options()
        options.add_argument("--disable-infobars")
        options.add_argument("--mute-audio")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument(f"--window-size={window_size}")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        if headless:
            options.add_argument("--headless=new")

        self.driver = webdriver.Chrome(options=options)
        self.page_url = page_url
        self.state_js = build_browser_state_js()
        self.accelerate = accelerate
        self._started = False
        self._duck_pressed = False
        self._default_acceleration = 0.001

        try:
            self.driver.get(page_url)
        except WebDriverException:
            # chrome://dino raises a navigation exception even though the page
            # still loads and the game is usable.
            pass

        time.sleep(1.0)
        self._body = self.driver.find_element("tag name", "body")
        self._ensure_runner_instance()
        default_accel = self.driver.execute_script("""
            Runner.instance_ = Runner.getInstance();
            var runnerAccel = (
                Runner.instance_.config && Runner.instance_.config.ACCELERATION
            );
            var globalAccel = Runner.config && Runner.config.ACCELERATION;
            return runnerAccel || globalAccel || 0.001;
            """)
        if default_accel is not None:
            self._default_acceleration = float(default_accel)
        self.set_acceleration(accelerate)

    def _ensure_runner_instance(self):
        self.driver.execute_script("Runner.instance_ = Runner.getInstance();")

    def set_acceleration(self, enabled: bool):
        value = self._default_acceleration if enabled else 0.0
        self.driver.execute_script(
            """
            Runner.instance_ = Runner.getInstance();
            if (Runner.config) {
                Runner.config.ACCELERATION = arguments[0];
            }
            if (Runner.instance_.config) {
                Runner.instance_.config.ACCELERATION = arguments[0];
            }
            """,
            value,
        )

    def start(self):
        self._ensure_runner_instance()
        self.set_duck(False)
        self.driver.execute_script("""
            Runner.instance_.playIntro();
            Runner.instance_.startGame();
            """)
        self._started = True

    def restart(self):
        self._ensure_runner_instance()
        self.set_duck(False)
        self.driver.execute_script("Runner.instance_.restart();")
        self.set_acceleration(self.accelerate)

    def jump(self):
        self.driver.execute_script(
            "Runner.instance_.tRex.startJump(Runner.instance_.currentSpeed);"
        )

    def set_duck(self, pressed: bool):
        if pressed == self._duck_pressed:
            return
        if pressed:
            ActionChains(self.driver).click(self._body).key_down(
                Keys.ARROW_DOWN
            ).perform()
        else:
            ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        self._duck_pressed = pressed

    def get_state(self) -> dict | None:
        try:
            self._ensure_runner_instance()
            return self.driver.execute_script(self.state_js)
        except JavascriptException:
            return None

    def get_score(self) -> int:
        state = self.get_state()
        if state is None:
            return 0
        return int(state.get("score", 0))

    def get_frame(self, obs_size: int = 84) -> np.ndarray:
        """Capture the current runner canvas as a normalised grayscale frame."""
        self._ensure_runner_instance()
        encoded = self.driver.execute_script(GET_CANVAS_JS)
        if encoded is None:
            raise RuntimeError("Runner canvas is not available.")
        png = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError("Failed to decode runner canvas frame.")
        frame = cv2.resize(frame, (obs_size, obs_size), interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32) / 255.0

    def close(self):
        try:
            self.set_duck(False)
        except JavascriptException:
            pass
        self.driver.quit()


class ChromeDinoFeatureEnv:
    """
    Feature-vector environment backed by a real Chrome Dino instance.

    The environment waits for distanceRan to advance after each action so PPO
    sees approximately one browser game frame per environment step.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        accelerate: bool = False,
        page_url: str = "chrome://dino",
        frame_timeout_sec: float = 0.25,
        poll_interval_sec: float = 0.002,
    ):
        self.game = ChromeDinoGame(
            page_url=page_url,
            headless=headless,
            accelerate=accelerate,
        )
        self.action_space = ActionSpace(3)
        self.reward_range = (-1.0, 0.1)
        self.frame_timeout_sec = frame_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.gametime_reward = 0.1
        self.gameover_penalty = -1.0
        self._started = False
        self._last_distance_ran = 0.0
        self._last_state: dict | None = None

    def _wait_for_state(
        self, *, require_playing: bool, prev_distance: float | None = None
    ):
        deadline = time.perf_counter() + self.frame_timeout_sec
        best_state = None

        while time.perf_counter() < deadline:
            state = self.game.get_state()
            if state is None:
                time.sleep(self.poll_interval_sec)
                continue

            best_state = state
            if state["crashed"]:
                return state

            if require_playing and not state["playing"]:
                time.sleep(self.poll_interval_sec)
                continue

            if prev_distance is None:
                return state

            if state.get("distanceRan", 0.0) > prev_distance:
                return state

            time.sleep(self.poll_interval_sec)

        if best_state is None:
            raise RuntimeError("Timed out while waiting for Chrome Dino state.")
        if require_playing and not best_state["playing"] and not best_state["crashed"]:
            raise RuntimeError(
                "Chrome Dino did not enter playing state before timeout."
            )
        return best_state

    def reset(self):
        if not self._started:
            self.game.start()
            self._started = True
        else:
            self.game.restart()

        state = self._wait_for_state(require_playing=True)
        self._last_state = state
        self._last_distance_ran = float(state.get("distanceRan", 0.0))
        return np.asarray(state["features"], dtype=np.float32)

    def step(self, action: int):
        assert action in self.action_space
        self.game.set_duck(action == 2)
        if action == 1:
            self.game.jump()

        state = self._wait_for_state(
            require_playing=False,
            prev_distance=self._last_distance_ran,
        )
        self._last_state = state
        self._last_distance_ran = float(
            state.get("distanceRan", self._last_distance_ran)
        )

        done = bool(state["crashed"])
        reward = self.gameover_penalty if done else self.gametime_reward
        return (
            np.asarray(state["features"], dtype=np.float32),
            reward,
            done,
            {"score": int(state["score"])},
        )

    def get_score(self) -> int:
        if self._last_state is not None:
            return int(self._last_state.get("score", 0))
        return self.game.get_score()

    def close(self):
        self.game.close()


class ChromeDinoImageEnv:
    """
    Frame-stacked browser Dino environment for image-based agents.

    Returns grayscale observations of shape (frame_stack, obs_size, obs_size).
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        accelerate: bool = False,
        page_url: str = "chrome://dino",
        frame_timeout_sec: float = 0.25,
        poll_interval_sec: float = 0.002,
        obs_size: int = 84,
        frame_stack: int = 4,
    ):
        self.game = ChromeDinoGame(
            page_url=page_url,
            headless=headless,
            accelerate=accelerate,
        )
        self.action_space = ActionSpace(3)
        self.frame_timeout_sec = frame_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.obs_size = obs_size
        self.frame_stack = frame_stack
        self.gametime_reward = 0.01
        self.gameover_penalty = -10.0
        self._started = False
        self._last_distance_ran = 0.0
        self._last_state: dict | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    def _wait_for_state(
        self, *, require_playing: bool, prev_distance: float | None = None
    ):
        deadline = time.perf_counter() + self.frame_timeout_sec
        best_state = None

        while time.perf_counter() < deadline:
            state = self.game.get_state()
            if state is None:
                time.sleep(self.poll_interval_sec)
                continue

            best_state = state
            if state["crashed"]:
                return state

            if require_playing and not state["playing"]:
                time.sleep(self.poll_interval_sec)
                continue

            if prev_distance is None:
                return state

            if state.get("distanceRan", 0.0) > prev_distance:
                return state

            time.sleep(self.poll_interval_sec)

        if best_state is None:
            raise RuntimeError("Timed out while waiting for Chrome Dino state.")
        if require_playing and not best_state["playing"] and not best_state["crashed"]:
            raise RuntimeError(
                "Chrome Dino did not enter playing state before timeout."
            )
        return best_state

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0).astype(np.float32)

    def reset(self):
        if not self._started:
            self.game.start()
            self._started = True
        else:
            self.game.restart()

        state = self._wait_for_state(require_playing=True)
        self._last_state = state
        self._last_distance_ran = float(state.get("distanceRan", 0.0))
        frame = self.game.get_frame(self.obs_size)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame.copy())
        return self._stacked_obs()

    def step(self, action: int):
        assert action in self.action_space
        self.game.set_duck(action == 2)
        if action == 1:
            self.game.jump()

        state = self._wait_for_state(
            require_playing=False,
            prev_distance=self._last_distance_ran,
        )
        self._last_state = state
        self._last_distance_ran = float(
            state.get("distanceRan", self._last_distance_ran)
        )
        self._frames.append(self.game.get_frame(self.obs_size))

        done = bool(state["crashed"])
        reward = self.gameover_penalty if done else self.gametime_reward
        return (
            self._stacked_obs(),
            reward,
            done,
            {"score": int(state["score"])},
        )

    def get_score(self) -> int:
        if self._last_state is not None:
            return int(self._last_state.get("score", 0))
        return self.game.get_score()

    def close(self):
        self.game.close()
