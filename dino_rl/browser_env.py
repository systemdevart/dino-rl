"""
ChromeDriver-backed Dino environment adapted from gym-chrome-dino.

The upstream project provides the basic Selenium/ChromeDriver control flow.
This adaptation keeps that idea but uses this repo's existing feature
contract, score handling, and Chrome 145+ Runner access pattern.
"""

from __future__ import annotations

import base64
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from selenium import webdriver
from selenium.common.exceptions import (
    JavascriptException,
    SessionNotCreatedException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from dino_rl.env import ActionSpace
from dino_rl.feature_contract import FEATURE_DIM, build_browser_state_js

# Re-aliasing Runner.instance_ inside each hot-loop JS payload (instead of a
# separate execute_script round-trip) keeps the call self-healing while saving
# one Selenium round-trip (~6ms) per call. ChromeDriver round-trips dominate the
# env step, so cutting redundant ones is the main throughput lever.
_RUNNER_ALIAS_JS = "Runner.instance_ = Runner.getInstance();\n"

GET_CANVAS_JS = """
Runner.instance_ = Runner.getInstance();
var canvas = document.getElementsByClassName('runner-canvas')[0];
return canvas ? canvas.toDataURL('image/png').substring(22) : null;
"""

_RECOVERABLE_BROWSER_ERROR_MARKERS = (
    "tab crashed",
    "invalid session id",
    "session deleted because of page crash",
    "disconnected",
    "max retries exceeded",
    "failed to establish a new connection",
    "connection refused",
    "web view not found",
    "target frame detached",
    "chrome not reachable",
)


def is_recoverable_browser_error(error: Exception) -> bool:
    """Return whether this Selenium failure is worth rebuilding the session for."""
    message = str(error).lower()
    return any(marker in message for marker in _RECOVERABLE_BROWSER_ERROR_MARKERS)


class ChromeDinoGame:
    """Thin controller around a real Chrome Dino instance."""

    def __init__(
        self,
        *,
        page_url: str = "chrome://dino",
        headless: bool = True,
        accelerate: bool = False,
        window_size: str = "800,600",
        startup_retries: int = 3,
        startup_retry_delay_sec: float = 1.0,
    ):
        self.page_url = page_url
        # Self-aliasing state poll: ensures Runner.instance_ in the same
        # round-trip as the state read, so the hot loop needs no separate
        # _ensure_runner_instance() call.
        self.state_js = _RUNNER_ALIAS_JS + build_browser_state_js()
        self.accelerate = accelerate
        self._started = False
        self._duck_pressed = False
        self._default_acceleration = 0.001
        self._window_size = window_size
        self._headless = headless
        self._startup_retries = startup_retries
        self._startup_retry_delay_sec = startup_retry_delay_sec
        self._profile_dir: str | None = None
        self._body = None
        self.driver = None

        self._launch_session()

    def _launch_session(self):
        last_error = None
        for attempt in range(1, self._startup_retries + 1):
            self._profile_dir = tempfile.mkdtemp(
                prefix="dino-chrome-profile-",
                dir="/tmp",
            )
            try:
                self.driver = webdriver.Chrome(
                    options=self._build_options(
                        headless=self._headless,
                        window_size=self._window_size,
                        profile_dir=self._profile_dir,
                    )
                )
                break
            except (SessionNotCreatedException, WebDriverException) as exc:
                last_error = exc
                self._cleanup_driver()
                self._cleanup_profile_dir()
                if attempt == self._startup_retries:
                    raise
                time.sleep(self._startup_retry_delay_sec * attempt)

        if self.driver is None:
            raise RuntimeError(
                "Failed to start Chrome Dino browser session."
            ) from last_error

        try:
            try:
                self.driver.get(self.page_url)
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
            self.set_acceleration(self.accelerate)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _build_options(*, headless: bool, window_size: str, profile_dir: str):
        options = Options()
        options.add_argument("--disable-infobars")
        options.add_argument("--mute-audio")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument(f"--window-size={window_size}")
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        if headless:
            options.add_argument("--headless=new")
        return options

    def _cleanup_driver(self):
        if self.driver is None:
            return
        try:
            self.driver.quit()
        except Exception:
            pass
        finally:
            self._body = None
            self.driver = None

    def _cleanup_profile_dir(self):
        if self._profile_dir is None:
            return
        shutil.rmtree(self._profile_dir, ignore_errors=True)
        self._profile_dir = None

    def _ensure_runner_instance(self):
        self.driver.execute_script("Runner.instance_ = Runner.getInstance();")

    def recover_session(self):
        self.close()
        self._started = False
        self._duck_pressed = False
        self._launch_session()

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

    def enable_deterministic(self):
        """Pause the game and install a controlled frame-stepper.

        The Chrome game normally runs in real time, so the ~tens of ms we spend
        on screenshot + round-trips + model inference let the game advance past
        the observed frame -- the action lands late and the observation/action
        desync. That makes the browser a much harder (and non-reproducible)
        environment than the discrete simulator, and breaks transfer of
        sim-trained policies.

        Here we stop() the game (which cancels its requestAnimationFrame loop and
        freezes distanceRan), suppress the auto-reschedule, and expose
        window.__envStep(action, n): it applies the action, then advances EXACTLY
        n game frames by driving Runner.update() with a fixed per-frame delta, and
        returns the state. Between calls the game is frozen, so our screenshot and
        inference cause zero drift -- a clean discrete step, like the sim.
        """
        self.driver.execute_script(
            """
            var r = Runner.instance_;
            r.stop();
            if (!r._detOrigSched) { r._detOrigSched = r.scheduleNextUpdate; }
            r.scheduleNextUpdate = function(){};   // suppress rAF re-arm
            // Virtual clock: the game reads time via performance.now() inside
            // getTimeStamp(); under real time the delta between our setting
            // r.time and update() reading now jitters (esp. under parallel CPU
            // load), so deltaTime != msPerFrame and the physics drift. Override
            // performance.now() to a controlled clock that advances EXACTLY one
            // frame per step -> deltaTime == msPerFrame exactly, fully
            // deterministic and parallel-safe (precise jump timing).
            if (typeof r._vclock !== 'number') { r._vclock = 1e6; }
            if (!performance._detPatched) {
                performance.now = function() { return Runner.instance_._vclock; };
                var _origDateNow = Date.now;
                Date.now = function() { return Runner.instance_._vclock; };
                performance._detPatched = true;
            }
            r.time = r._vclock;
            window.__getState = function() { %s };
            window.__envStep = function(action, n) {
                var r = Runner.instance_;
                var t = r.tRex;
                // Apply the action with EXACTLY the sim's (and the real key
                // handler's) semantics. The old code called setDuck(action===2)
                // unconditionally, which put the trex into DUCKING status
                // MID-AIR -- a (jumping=1, ducking=1) state that never exists in
                // the sim (it routes down-while-airborne to speed-drop), so
                // sim-trained policies went out-of-distribution and died.
                if (action === 1) {
                    if (!t.jumping) {
                        if (t.ducking && t.setDuck) { t.setDuck(false); }
                        t.startJump(r.currentSpeed);
                    }
                } else if (action === 2) {
                    if (t.jumping) {
                        t.speedDrop = true;              // sim: speed_drop, not duck
                    } else if (t.setDuck && !t.ducking) {
                        t.setDuck(true);
                    }
                } else {
                    if (!t.jumping && t.ducking && t.setDuck) { t.setDuck(false); }
                }
                r.playing = true;
                for (var i = 0; i < n; i++) {
                    r._vclock += r.msPerFrame;   // advance exactly one frame
                    r.update();                  // deltaTime == msPerFrame
                }
                return window.__getState();
            };
            return true;
            """ % self.state_js
        )
        self._deterministic = True

    def env_step(self, action: int, n_frames: int) -> dict | None:
        """One deterministic step: apply action, advance n_frames, return state."""
        try:
            return self.driver.execute_script(
                "return window.__envStep(arguments[0], arguments[1]);",
                int(action), int(n_frames),
            )
        except JavascriptException:
            return None

    def disable_deterministic(self):
        """Restore the normal real-time loop (needed before restart()/quit)."""
        try:
            self.driver.execute_script(
                """
                var r = Runner.instance_;
                if (r && r._detOrigSched) { r.scheduleNextUpdate = r._detOrigSched; }
                """
            )
        except JavascriptException:
            pass
        self._deterministic = False

    def restart_deterministic(self):
        """JS-only restart for deterministic mode, with a reload fallback.

        The normal restart() wraps the JS in ActionChains set_duck +
        set_acceleration, which misbehave while paused. This restores the normal
        update loop and restarts in a single JS call, then re-installs the
        frame-stepper. Runner.restart() occasionally throws an internal assertion
        for certain crash/obstacle states, so on failure we reload the page and
        start fresh (slower, but bulletproof).
        """
        try:
            self.driver.execute_script(
                """
                var r = Runner.instance_ = Runner.getInstance();
                if (r._detOrigSched) { r.scheduleNextUpdate = r._detOrigSched; }
                r.restart();
                """
            )
            self._duck_pressed = False
            self.set_acceleration(self.accelerate)
            self.enable_deterministic()
            return
        except JavascriptException:
            pass
        # Fallback: reload chrome://dino and start a fresh game.
        try:
            self.driver.get(self.page_url)
        except WebDriverException:
            pass
        time.sleep(0.4)
        self._body = self.driver.find_element("tag name", "body")
        self._ensure_runner_instance()
        self._duck_pressed = False
        self.driver.execute_script(
            "Runner.instance_.playIntro(); Runner.instance_.startGame();"
        )
        self.set_acceleration(self.accelerate)
        self.enable_deterministic()

    def get_state(self) -> dict | None:
        try:
            # state_js self-aliases Runner.instance_, so no separate
            # _ensure_runner_instance() round-trip is needed here.
            return self.driver.execute_script(self.state_js)
        except JavascriptException:
            return None

    def get_score(self) -> int:
        state = self.get_state()
        if state is None:
            return 0
        return int(state.get("score", 0))

    def get_frame(self, obs_size: int | None = 84) -> np.ndarray:
        """Capture the current runner canvas as a grayscale frame."""
        # GET_CANVAS_JS self-aliases Runner.instance_, so no separate
        # _ensure_runner_instance() round-trip is needed here.
        encoded = self.driver.execute_script(GET_CANVAS_JS)
        if encoded is None:
            raise RuntimeError("Runner canvas is not available.")
        png = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError("Failed to decode runner canvas frame.")
        if obs_size is None:
            return frame
        frame = cv2.resize(frame, (obs_size, obs_size), interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32) / 255.0

    def close(self):
        try:
            self.set_duck(False)
        except Exception:
            pass
        self._cleanup_driver()
        self._cleanup_profile_dir()
        self._duck_pressed = False


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

    def _cached_obs(self) -> np.ndarray:
        if self._last_state is not None:
            return np.asarray(self._last_state["features"], dtype=np.float32)
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    def _recover_browser_session(self, exc: Exception, *, context: str):
        if not is_recoverable_browser_error(exc):
            raise exc
        print(f"[browser] Recovering Chrome Dino feature env after {context}: {exc}")
        self.game.recover_session()
        self._started = False
        self._last_distance_ran = 0.0
        self._last_state = None

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

    def reset(self, _allow_recover: bool = True):
        try:
            if not self._started:
                self.game.start()
                self._started = True
            else:
                self.game.restart()

            state = self._wait_for_state(require_playing=True)
            self._last_state = state
            self._last_distance_ran = float(state.get("distanceRan", 0.0))
            return np.asarray(state["features"], dtype=np.float32)
        except Exception as exc:
            if not _allow_recover or not is_recoverable_browser_error(exc):
                raise
            self._recover_browser_session(exc, context="feature reset")
            return self.reset(_allow_recover=False)

    def step(self, action: int):
        assert action in self.action_space
        fallback_obs = self._cached_obs()
        fallback_score = (
            int(self._last_state.get("score", 0)) if self._last_state else 0
        )

        try:
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
        except Exception as exc:
            self._recover_browser_session(
                exc,
                context=f"feature step(action={action})",
            )
            return (
                fallback_obs,
                self.gameover_penalty,
                True,
                {"score": fallback_score, "browser_recovered": True},
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
        action_repeat: int = 4,
        crop_top_ratio: float = 0.15,
        crop_bottom_ratio: float = 0.98,
        score_mask_left_ratio: float = 0.82,
        score_mask_height_ratio: float = 0.28,
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
        self.action_repeat = action_repeat
        self.distance_reward_scale = 0.02
        self.gameover_penalty = -1.0
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.score_mask_left_ratio = score_mask_left_ratio
        self.score_mask_height_ratio = score_mask_height_ratio
        self._started = False
        self._last_distance_ran = 0.0
        self._last_state: dict | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    def _cached_obs(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(
                (self.frame_stack, self.obs_size, self.obs_size),
                dtype=np.float32,
            )
        frames = [frame.copy() for frame in self._frames]
        while len(frames) < self.frame_stack:
            frames.append(frames[-1].copy())
        return np.stack(frames, axis=0).astype(np.float32)

    def _recover_browser_session(self, exc: Exception, *, context: str):
        if not is_recoverable_browser_error(exc):
            raise exc
        print(f"[browser] Recovering Chrome Dino image env after {context}: {exc}")
        self.game.recover_session()
        self._started = False
        self._last_distance_ran = 0.0
        self._last_state = None
        self._frames.clear()

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

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop to the gameplay strip and suppress the score HUD."""
        height, width = frame.shape
        top = min(max(int(height * self.crop_top_ratio), 0), height - 1)
        bottom = min(max(int(height * self.crop_bottom_ratio), top + 1), height)
        cropped = frame[top:bottom].copy()

        mask_x = min(
            max(int(cropped.shape[1] * self.score_mask_left_ratio), 0),
            cropped.shape[1],
        )
        mask_h = min(
            max(int(cropped.shape[0] * self.score_mask_height_ratio), 1),
            cropped.shape[0],
        )
        cropped[:mask_h, mask_x:] = 0

        resized = cv2.resize(
            cropped,
            (self.obs_size, self.obs_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32) / 255.0

    def _capture_frame(self) -> np.ndarray:
        return self._preprocess_frame(self.game.get_frame(obs_size=None))

    @staticmethod
    def _pool_recent_frames(frames: list[np.ndarray]) -> np.ndarray:
        if len(frames) >= 2:
            return np.maximum(frames[-2], frames[-1])
        return frames[-1]

    def reset(self, _allow_recover: bool = True):
        try:
            if not self._started:
                self.game.start()
                self._started = True
            else:
                self.game.restart()

            state = self._wait_for_state(require_playing=True)
            self._last_state = state
            self._last_distance_ran = float(state.get("distanceRan", 0.0))
            frame = self._capture_frame()
            self._frames.clear()
            for _ in range(self.frame_stack):
                self._frames.append(frame.copy())
            return self._stacked_obs()
        except Exception as exc:
            if not _allow_recover or not is_recoverable_browser_error(exc):
                raise
            self._recover_browser_session(exc, context="image reset")
            return self.reset(_allow_recover=False)

    def step(self, action: int):
        assert action in self.action_space
        fallback_obs = self._cached_obs()
        total_reward = 0.0
        recent_frames: list[np.ndarray] = []
        score = int(self._last_state.get("score", 0)) if self._last_state else 0
        done = False

        try:
            for repeat_idx in range(self.action_repeat):
                self.game.set_duck(action == 2)
                if action == 1 and repeat_idx == 0:
                    self.game.jump()

                prev_distance = self._last_distance_ran
                state = self._wait_for_state(
                    require_playing=False,
                    prev_distance=prev_distance,
                )
                self._last_state = state
                current_distance = float(state.get("distanceRan", prev_distance))
                distance_delta = max(current_distance - prev_distance, 0.0)
                self._last_distance_ran = current_distance
                score = int(state["score"])
                recent_frames.append(self._capture_frame())
                total_reward += self.distance_reward_scale * distance_delta

                done = bool(state["crashed"])
                if done:
                    total_reward += self.gameover_penalty
                    break

            if not recent_frames:
                recent_frames.append(self._capture_frame())
            self._frames.append(self._pool_recent_frames(recent_frames))

            return (
                self._stacked_obs(),
                total_reward,
                done,
                {"score": score},
            )
        except Exception as exc:
            self._recover_browser_session(
                exc,
                context=f"image step(action={action})",
            )
            return (
                fallback_obs,
                self.gameover_penalty,
                True,
                {"score": score, "browser_recovered": True},
            )

    def get_score(self) -> int:
        if self._last_state is not None:
            return int(self._last_state.get("score", 0))
        return self.game.get_score()

    def close(self):
        self.game.close()


class VecChromeDinoImageEnv:
    """
    Run several ChromeDinoImageEnv workers in parallel threads.

    Browser steps are I/O-bound: each Selenium call blocks on a ChromeDriver
    HTTP round-trip and releases the GIL, so worker threads give real
    parallelism and let a single PPO learner consume many browser sessions at
    once. Each worker owns its own Chrome + ChromeDriver, and a thread only ever
    touches its own worker, so there is no shared-driver concurrency.

    step() auto-resets any worker whose episode just ended (returning the
    post-reset observation), so rollout collection never stalls on one env.
    Per-worker browser recovery is handled inside ChromeDinoImageEnv, so a crash
    in one worker does not take down the others.
    """

    def __init__(self, num_envs: int, **env_kwargs):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        self.num_envs = num_envs
        self._pool = ThreadPoolExecutor(max_workers=num_envs)

        # Launch all browser sessions concurrently to cut startup time, and
        # clean up any that did start if one fails to launch.
        futures = [
            self._pool.submit(ChromeDinoImageEnv, **env_kwargs)
            for _ in range(num_envs)
        ]
        self.envs: list[ChromeDinoImageEnv] = []
        launch_error: Exception | None = None
        for future in futures:
            try:
                self.envs.append(future.result())
            except Exception as exc:  # noqa: BLE001 - re-raised after cleanup
                launch_error = exc
        if launch_error is not None:
            self.close()
            raise launch_error

        self.action_space = self.envs[0].action_space
        self.obs_shape = (
            self.envs[0].frame_stack,
            self.envs[0].obs_size,
            self.envs[0].obs_size,
        )

    def reset(self) -> np.ndarray:
        obs = list(self._pool.map(lambda env: env.reset(), self.envs))
        return np.stack(obs, axis=0)

    def step(self, actions):
        """Step every worker in parallel; auto-reset finished workers.

        Returns batched (obs[N], rewards[N], dones[N], infos[list]).
        """
        results = list(
            self._pool.map(
                lambda i: self.envs[i].step(int(actions[i])),
                range(self.num_envs),
            )
        )
        obs = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([bool(r[2]) for r in results], dtype=bool)
        infos = [r[3] for r in results]

        # Auto-reset finished workers in parallel so the next tick starts fresh.
        reset_ids = [i for i, done in enumerate(dones) if done]
        if reset_ids:
            reset_obs = list(
                self._pool.map(lambda i: self.envs[i].reset(), reset_ids)
            )
            for i, frame in zip(reset_ids, reset_obs):
                obs[i] = frame

        return np.stack(obs, axis=0), rewards, dones, infos

    def get_scores(self) -> np.ndarray:
        """Current per-worker score (cached, no browser round-trip)."""
        return np.array([env.get_score() for env in self.envs], dtype=np.int64)

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self._pool.shutdown(wait=True)
