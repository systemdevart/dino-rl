"""
Play Chrome Dino game in the browser using the trained DQN model.

Uses Selenium to open Chrome to chrome://dino and controls the game
entirely via JavaScript (state extraction + jump commands).
The model handles the real game's increasing speed (no cheats).

Requirements:
    pip install selenium torch numpy

Usage:
    python play_browser.py                    # Run with default model
    python play_browser.py --model path.pth   # Run with specific model
    python play_browser.py --games 5          # Play N games then quit
"""
import argparse
import time

import torch
import torch.nn as nn

from selenium import webdriver
from selenium.common.exceptions import WebDriverException


FEATURE_DIM = 8  # Includes current speed


class DuelingDQN(nn.Module):
    """Same architecture as training - must match exactly."""
    def __init__(self, input_dim, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_size)

    def forward(self, x):
        feat = self.feature(x)
        value = self.value(feat)
        advantage = self.advantage(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# JavaScript injected into the page to extract game state features.
# Returns an object with 8 normalized features matching the training env,
# plus crash/score/playing status.
# Uses Runner.instance_ which we alias from Runner.getInstance() at setup
# (Chrome 145+ no longer exposes Runner.instance_ directly).
GET_STATE_JS = """
return (function() {
    var r = Runner.instance_;
    if (!r || !r.tRex) return null;

    var tRex = r.tRex;
    var obstacles = r.horizon.obstacles;
    var canvasWidth = r.dimensions.WIDTH || 600;
    var maxSpeed = 13.0;

    // Filter obstacles that are ahead of the dino AND at a jumpable height.
    // High-flying pterodactyls (bottom edge above dino's head) can be run
    // under safely, so we exclude them.
    var dinoTop = tRex.groundYPos || 93;
    var ahead = [];
    for (var i = 0; i < obstacles.length; i++) {
        var o = obstacles[i];
        if (o.yPos + o.typeConfig.height <= dinoTop) continue;
        if (o.xPos + o.width > tRex.xPos) {
            ahead.push({x: o.xPos, w: o.width, h: o.typeConfig.height});
        }
    }
    ahead.sort(function(a, b) { return a.x - b.x; });

    // Feature 0-2: nearest obstacle distance, width, height
    var dist1, w1, h1;
    if (ahead.length >= 1) {
        dist1 = (ahead[0].x - tRex.xPos) / canvasWidth;
        w1 = ahead[0].w / 51.0;
        h1 = ahead[0].h / 50.0;
    } else {
        dist1 = 1.0; w1 = 0.0; h1 = 0.0;
    }

    // Feature 6: distance to 2nd nearest obstacle
    var dist2 = (ahead.length >= 2) ?
        (ahead[1].x - tRex.xPos) / canvasWidth : 1.0;

    // Feature 3: dino height above ground
    var groundY = tRex.groundYPos || 93;
    var dinoHeight = Math.max(0, groundY - tRex.yPos) / 90.0;

    // Feature 4: dino vertical velocity
    var dinoVel = tRex.jumping ? (tRex.jumpVelocity / 10.0) : 0.0;

    // Feature 5: is jumping
    var jumping = tRex.jumping ? 1.0 : 0.0;

    // Feature 7: current game speed (normalized by max speed)
    var speed = r.currentSpeed / maxSpeed;

    // Score
    var scoreStr = r.distanceMeter.digits.join('');

    return {
        features: [dist1, w1, h1, dinoHeight, dinoVel, jumping, dist2, speed],
        crashed: r.crashed,
        score: parseInt(scoreStr) || 0,
        playing: r.playing
    };
})();
"""


def load_model(weight_path):
    """Load trained DuelingDQN model."""
    device = torch.device("cpu")
    model = DuelingDQN(FEATURE_DIM, 2).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded model from {weight_path}")
    return model, device


def get_action(model, device, features):
    """Run inference to get best action."""
    with torch.no_grad():
        state = torch.FloatTensor(features).unsqueeze(0).to(device)
        q_values = model(state)
        return q_values.argmax(dim=1).item()


def setup_browser(headless=False):
    """Launch Chrome and navigate to the dino game."""
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    if headless:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get('chrome://dino')
    except WebDriverException:
        pass  # Expected - Chrome shows the dino game despite the "error"

    time.sleep(1)

    # Chrome 145+ stores the instance in a closure variable accessible only
    # via Runner.getInstance(). Alias it to Runner.instance_ for our JS code.
    driver.execute_script("Runner.instance_ = Runner.getInstance();")

    return driver


def play(driver, model, device, max_games=None):
    """Main game loop: extract state, pick action, control via JS."""

    # Start the first game via JavaScript (works in both headed and headless)
    driver.execute_script("Runner.instance_.playIntro();")
    time.sleep(0.5)
    driver.execute_script("Runner.instance_.startGame();")
    time.sleep(0.3)

    print("=" * 60, flush=True)
    print("  DQN Agent is playing Chrome Dino!", flush=True)
    print("  (with real acceleration - no cheats)", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    best_score = 0
    games = 0
    steps = 0

    try:
        while True:
            state = driver.execute_script(GET_STATE_JS)

            if state is None:
                time.sleep(0.05)
                continue

            if state['crashed']:
                games += 1
                score = state['score']
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
                driver.execute_script("Runner.instance_.restart();")
                time.sleep(0.3)
                continue

            if not state['playing']:
                time.sleep(0.05)
                continue

            # Get model's action and execute immediately
            features = state['features']
            action = get_action(model, device, features)
            steps += 1

            if action == 1:
                driver.execute_script(
                    "Runner.instance_.tRex.startJump(Runner.instance_.currentSpeed);"
                )

    except KeyboardInterrupt:
        print(f"\n\nStopped by user. Games: {games}, Best score: {best_score}")


def main():
    parser = argparse.ArgumentParser(description="Play Chrome Dino with trained DQN")
    parser.add_argument(
        "--model", default="models/dino_runner.pth",
        help="Path to trained model weights (default: models/dino_runner.pth)"
    )
    parser.add_argument(
        "--games", type=int, default=None,
        help="Number of games to play (default: unlimited)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run Chrome in headless mode (no GUI window)"
    )
    args = parser.parse_args()

    model, device = load_model(args.model)
    driver = setup_browser(headless=args.headless)

    try:
        play(driver, model, device, max_games=args.games)
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == '__main__':
    main()
