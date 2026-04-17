"""
Play Chrome Dino game in the browser using a trained DQN or PPO policy.

Uses Selenium to open Chrome to chrome://dino and controls the game
entirely via JavaScript (state extraction + jump commands).
The model handles the real game's increasing speed (no cheats).

Requirements:
    pip install selenium torch numpy

Usage:
    python -m dino_rl.play_browser                      # Run with default model
    python -m dino_rl.play_browser --model path.pth     # Run with specific model
    python -m dino_rl.play_browser --algo ppo --games 5 # Force PPO checkpoint
"""

import argparse
import time

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from dino_rl.feature_contract import build_browser_state_js
from dino_rl.policy_loader import load_policy
from dino_rl.policy_paths import default_policy_path

GET_STATE_JS = build_browser_state_js()
JUMP_JS = "Runner.instance_.tRex.startJump(Runner.instance_.currentSpeed);"


def setup_browser(headless=False):
    """Launch Chrome and navigate to the dino game."""
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    if headless:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get("chrome://dino")
    except WebDriverException:
        pass  # Expected - Chrome shows the dino game despite the "error"

    time.sleep(1)

    # Chrome 145+ stores the instance in a closure variable accessible only
    # via Runner.getInstance(). Alias it to Runner.instance_ for our JS code.
    driver.execute_script("Runner.instance_ = Runner.getInstance();")

    return driver


def set_duck(driver, body, pressed, duck_pressed):
    """Press or release ArrowDown only when the state changes."""
    if pressed == duck_pressed:
        return duck_pressed
    if pressed:
        ActionChains(driver).click(body).key_down(Keys.ARROW_DOWN).perform()
    else:
        ActionChains(driver).key_up(Keys.ARROW_DOWN).perform()
    return pressed


def play(driver, policy, max_games=None):
    """Main game loop: extract state, pick action, control via JS."""

    # Start the first game via JavaScript (works in both headed and headless)
    driver.execute_script("Runner.instance_.playIntro();")
    time.sleep(0.5)
    driver.execute_script("Runner.instance_.startGame();")
    time.sleep(0.3)

    print("=" * 60, flush=True)
    print(f"  {policy.algo.upper()} policy is playing Chrome Dino!", flush=True)
    print("  (with real acceleration - no cheats)", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    best_score = 0
    games = 0
    steps = 0
    duck_pressed = False
    body = driver.find_element("tag name", "body")

    try:
        while True:
            state = driver.execute_script(GET_STATE_JS)

            if state is None:
                time.sleep(0.05)
                continue

            if state["crashed"]:
                games += 1
                score = state["score"]
                if score > best_score:
                    best_score = score
                print(
                    f"  Game {games:3d} | Score: {score:6d} | "
                    f"Best: {best_score:6d} | Steps: {steps}",
                    flush=True,
                )
                steps = 0

                if max_games and games >= max_games:
                    print(f"\nFinished {max_games} games. Best score: {best_score}")
                    break

                # Restart via JS
                time.sleep(0.5)
                duck_pressed = set_duck(driver, body, False, duck_pressed)
                driver.execute_script("Runner.instance_.restart();")
                time.sleep(0.3)
                continue

            if not state["playing"]:
                time.sleep(0.05)
                continue

            # Get model's action and execute immediately
            features = state["features"]
            action = policy.act(features)
            steps += 1

            duck_pressed = set_duck(driver, body, action == 2, duck_pressed)
            if action == 1:
                driver.execute_script(JUMP_JS)

    except KeyboardInterrupt:
        print(f"\n\nStopped by user. Games: {games}, Best score: {best_score}")
    finally:
        set_duck(driver, body, False, duck_pressed)


def main():
    parser = argparse.ArgumentParser(
        description="Play Chrome Dino with a trained DQN or PPO policy"
    )
    parser.add_argument(
        "--model",
        default=default_policy_path(),
        help=(
            "Path to trained model weights "
            "(defaults to checkpoints/dino_ppo_best.pth when present, "
            "otherwise checkpoints/dino_runner.pth)"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of games to play (default: unlimited)",
    )
    parser.add_argument(
        "--algo",
        choices=("auto", "dqn", "ppo"),
        default="auto",
        help="Checkpoint type (default: auto-detect from checkpoint contents)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (no GUI window)",
    )
    args = parser.parse_args()

    policy = load_policy(args.model, algo=args.algo, device="cpu")
    print(f"Loaded {policy.algo.upper()} policy from {args.model}")
    driver = setup_browser(headless=args.headless)

    try:
        play(driver, policy, max_games=args.games)
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    main()
