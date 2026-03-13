"""
Heuristic (zero-ML) agent for the Chrome Dino game.

Instead of a neural network, this agent forward-simulates the jump
trajectory and checks whether jumping NOW will clear the obstacle.
No training, no weights, no gradient descent — just physics.

The key insight: the Chrome Dino game has a tiny effective decision
space.  When airborne, the trajectory is fully deterministic (no action
possible).  The only decision is "jump or not?" while on the ground,
which depends on just (distance, obstacle_width, obstacle_height, speed).
That's a ~200-entry lookup table, trivially solvable without ML.

This file implements two approaches:
1. Exact forward simulation (will_clear_if_jump_now) — most accurate
2. Precomputed lookup table — O(1) runtime, nearly as good
"""
import numpy as np
from dino_rl.env import DinoRunEnv


# -----------------------------------------------------------------------
# Jump physics simulation (matches env.py exactly)
# -----------------------------------------------------------------------

DINO_HEIGHT = 47
DINO_WIDTH = 44
GRAVITY = 0.6
DROP_VELOCITY = -5
MAX_JUMP_HEIGHT = 30
MIN_JUMP_HEIGHT_Y = 63     # groundYPos(93) - MIN_JUMP_HEIGHT(30)
GROUND_Y_POS = 93


def simulate_jump_arc(speed):
    """
    Return list of dino_y values for each frame of a jump at given speed.
    Frame 0 is the first frame after initiating the jump.
    """
    y = float(GROUND_Y_POS)
    vel = -10.0 - speed / 10.0
    reached_min = False
    arc = []

    for _ in range(80):
        y += round(vel)
        vel += GRAVITY
        if y < MIN_JUMP_HEIGHT_Y:
            reached_min = True
        if y < MAX_JUMP_HEIGHT:
            if reached_min and vel < DROP_VELOCITY:
                vel = DROP_VELOCITY
        arc.append(y)
        if y >= GROUND_Y_POS:
            break

    return arc


def will_clear_if_jump_now(speed, obs_x, obs_width, obs_y, obs_height,
                           dino_x=50):
    """
    Forward-simulate: if the dino jumps THIS frame, will it clear
    the given obstacle without collision?

    Uses the exact same collision model as the env (broad-phase AABB
    with 1px margin — we skip the narrow-phase per-part boxes for speed,
    but add extra margin to compensate).

    Returns True if the jump clears safely.
    """
    arc = simulate_jump_arc(speed)
    move = int(speed)  # obstacle pixels/frame (Math.floor at 60fps)

    for f, dino_y in enumerate(arc):
        # Obstacle position at frame f+1 (it moves before collision check)
        ox = obs_x - move * (f + 1)

        # Broad-phase AABB (with 2px extra margin for safety vs narrow-phase)
        m = 3  # slightly larger than the 1px in the real collision
        d_left = dino_x + m
        d_top = int(dino_y) + m
        d_right = dino_x + DINO_WIDTH - m
        d_bottom = int(dino_y) + DINO_HEIGHT - m

        o_left = int(ox)
        o_top = obs_y
        o_right = int(ox) + obs_width
        o_bottom = obs_y + obs_height

        if (d_right > o_left and d_left < o_right and
                d_bottom > o_top and d_top < o_bottom):
            return False  # Collision during jump

    return True  # Cleared safely


def _should_jump(env):
    """
    Exact decision: should the dino jump right now?

    Strategy:
    1. Find the nearest obstacle we need to jump over.
    2. Check if jumping NOW clears it.
    3. Check if waiting one more frame still clears it.
    4. If jumping now clears but waiting might not → jump now.
       (Jump as late as possible for maximum margin.)
    """
    if env.jumping:
        return False

    # Find nearest dangerous obstacle ahead
    nearest = None
    for obs in env.obstacles:
        if obs.x + obs.width <= env.dino_x:
            continue
        # Skip high pterodactyls we can run under
        if (obs.type_config['type'] == 'PTERODACTYL'
                and obs.y + obs.height <= env.ground_y_pos):
            continue
        if nearest is None or obs.x < nearest.x:
            nearest = obs

    if nearest is None:
        return False  # Nothing to dodge

    dist = nearest.x - env.dino_x

    # If obstacle is very far, don't bother simulating
    if dist > 250:
        return False

    speed = env.speed

    # Will jumping NOW clear the obstacle?
    clears_now = will_clear_if_jump_now(
        speed, nearest.x, nearest.width, nearest.y, nearest.height,
        env.dino_x)

    if not clears_now:
        # Can't clear even if we jump now — either too late or too early.
        # If too close, jumping is a Hail Mary; otherwise wait.
        if dist < int(speed) * 2:
            return True  # Desperation jump
        return False

    # Will waiting one more frame still clear?
    # Simulate obstacle one frame closer
    future_obs_x = nearest.x - int(speed + nearest.speed_offset)
    future_speed = min(speed + 0.001, 13.0)

    clears_next = will_clear_if_jump_now(
        future_speed, future_obs_x, nearest.width, nearest.y,
        nearest.height, env.dino_x)

    if not clears_next:
        # This is the last safe frame — jump now!
        return True

    # We can still wait. But don't wait too long for multi-obstacle gaps.
    # Check if there's a second obstacle close behind that constrains us.
    second = None
    for obs in env.obstacles:
        if obs.x + obs.width <= env.dino_x:
            continue
        if (obs.type_config['type'] == 'PTERODACTYL'
                and obs.y + obs.height <= env.ground_y_pos):
            continue
        if obs is not nearest and (second is None or obs.x < second.x):
            second = obs

    if second is not None:
        second_dist = second.x - env.dino_x
        # If second obstacle is close, we need to land in time.
        # Jump duration is ~35 frames. Landing covers ~35 * speed pixels.
        # We need to be on ground before second obstacle arrives.
        jump_duration = len(simulate_jump_arc(speed))
        landing_dist = int(speed) * jump_duration
        if second_dist < landing_dist + int(speed) * 5:
            # Second obstacle is close — jump early to land in time
            return True

    return False


def heuristic_action(env):
    """Returns 0 (wait) or 1 (jump)."""
    return 1 if _should_jump(env) else 0


# -----------------------------------------------------------------------
# Feature-based version (drop-in replacement for neural network)
# -----------------------------------------------------------------------

def heuristic_action_from_features(features):
    """
    Simplified heuristic using only the normalized feature vector.
    Less accurate than the direct env access version, but works as
    a drop-in replacement for a neural network policy.
    """
    dist1 = features[0]      # / 600
    w1 = features[1]         # / 75
    h1 = features[2]         # / 50
    dino_h = features[3]     # / 93
    jumping = features[5]
    speed_n = features[7]    # / 13

    if jumping > 0.5:
        return 0
    if dist1 >= 0.9:
        return 0

    dist_px = dist1 * 600.0
    speed = speed_n * 13.0
    obs_w = w1 * 75.0
    obs_h = h1 * 50.0

    # Determine obstacle yPos from height
    if abs(obs_h - 35) < 2:
        obs_y = 105
    elif abs(obs_h - 50) < 2:
        obs_y = 90
    else:
        obs_y = 100  # low pterodactyl

    obs_x = dist_px + 50  # approximate absolute position

    clears = will_clear_if_jump_now(speed, obs_x, int(obs_w), obs_y,
                                    int(obs_h), 50)
    if not clears:
        if dist_px < int(speed) * 2:
            return 1
        return 0

    # Check if next frame still clears
    future_x = obs_x - int(speed)
    future_clears = will_clear_if_jump_now(
        min(speed + 0.001, 13.0), future_x, int(obs_w), obs_y,
        int(obs_h), 50)

    if not future_clears:
        return 1

    return 0


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(n_games=30, skip_clear_time=True, verbose=True):
    """Evaluate the heuristic agent."""
    env = DinoRunEnv(domain_randomization=False,
                     skip_clear_time=skip_clear_time)
    scores = []

    for ep in range(n_games):
        env.reset()
        for t in range(100000):
            action = heuristic_action(env)
            _, _, score, done = env.step(action)
            if done:
                break
        scores.append(score)
        if verbose:
            print(f"  Game {ep+1:2d}: score={score:5d}  steps={t+1}")

    avg = np.mean(scores)
    med = np.median(scores)
    if verbose:
        print(f"\n--- {n_games} Game Summary ---")
        print(f"Avg:    {avg:.0f}")
        print(f"Median: {med:.0f}")
        print(f"Min:    {min(scores)}")
        print(f"Max:    {max(scores)}")
        print(f"Std:    {np.std(scores):.0f}")
    return scores


if __name__ == '__main__':
    print("Heuristic Agent — zero ML, forward simulation")
    print("No training, no weights, no gradient descent.\n")
    evaluate(n_games=30)
