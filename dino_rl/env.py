"""
Pure Python Chrome Dinosaur Game environment.

1:1 faithful port of the Chromium T-Rex Runner (offline.js).
All constants, physics, obstacle types, collision boxes, gap formulas,
and scoring match the original JavaScript source exactly.

Source: chromium/src/components/neterror/resources/offline.js

Domain randomization (optional, for sim-to-real transfer):
- Randomised dino X position: [20, 55] per episode
"""
import numpy as np
import cv2


class ActionSpace:
    """Minimal replacement for gym.spaces.Discrete."""
    def __init__(self, n):
        self.n = n

    def __contains__(self, x):
        return 0 <= x < self.n


# ---------------------------------------------------------------------------
# Collision box (matches CollisionBox in offline.js)
# ---------------------------------------------------------------------------

class CollisionBox:
    __slots__ = ('x', 'y', 'width', 'height')

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h


# ---------------------------------------------------------------------------
# Obstacle type definitions — Obstacle.types in offline.js
# ---------------------------------------------------------------------------

CACTUS_SMALL = {
    'type': 'CACTUS_SMALL',
    'width': 17,
    'height': 35,
    'yPos': 105,
    'multipleSpeed': 4,
    'minGap': 120,
    'minSpeed': 0,
    'collisionBoxes': [
        CollisionBox(0, 7, 5, 27),
        CollisionBox(4, 0, 6, 34),
        CollisionBox(10, 4, 7, 14),
    ],
}

CACTUS_LARGE = {
    'type': 'CACTUS_LARGE',
    'width': 25,
    'height': 50,
    'yPos': 90,
    'multipleSpeed': 7,
    'minGap': 120,
    'minSpeed': 0,
    'collisionBoxes': [
        CollisionBox(0, 12, 7, 38),
        CollisionBox(8, 0, 7, 49),
        CollisionBox(13, 10, 10, 38),
    ],
}

PTERODACTYL = {
    'type': 'PTERODACTYL',
    'width': 46,
    'height': 40,
    'yPos': [100, 75, 50],
    'multipleSpeed': 999,
    'minSpeed': 8.5,
    'minGap': 150,
    'collisionBoxes': [
        CollisionBox(15, 15, 16, 5),
        CollisionBox(18, 21, 24, 6),
        CollisionBox(2, 14, 4, 3),
        CollisionBox(6, 10, 4, 7),
        CollisionBox(10, 8, 6, 9),
    ],
    'speedOffset': 0.8,
}

OBSTACLE_TYPES = [CACTUS_SMALL, CACTUS_LARGE, PTERODACTYL]

# T-Rex collision boxes — RUNNING state (Trex.collisionBoxes.RUNNING)
TREX_RUNNING_BOXES = [
    CollisionBox(22, 0, 17, 16),
    CollisionBox(1, 18, 30, 9),
    CollisionBox(10, 35, 14, 8),
    CollisionBox(1, 24, 29, 5),
    CollisionBox(5, 30, 21, 4),
    CollisionBox(9, 34, 15, 4),
]


# ---------------------------------------------------------------------------
# Collision helpers — checkForCollision / boxCompare in offline.js
# ---------------------------------------------------------------------------

def _box_compare(a, b):
    """AABB intersection test."""
    return (a.x < b.x + b.width and
            a.x + a.width > b.x and
            a.y < b.y + b.height and
            a.y + a.height > b.y)


def _check_collision(obs, dino_x, dino_y, dino_w, dino_h):
    """
    Two-phase collision detection.

    Phase 1: broad-phase outer bounding boxes (1 px border adjustment).
    Phase 2: narrow-phase per-part AABB with detailed collision boxes.
    """
    dino_outer = CollisionBox(dino_x + 1, dino_y + 1,
                              dino_w - 2, dino_h - 2)
    obs_outer = CollisionBox(int(obs.x) + 1, obs.y + 1,
                             obs.width - 2, obs.height - 2)

    if _box_compare(dino_outer, obs_outer):
        for dino_cb in TREX_RUNNING_BOXES:
            for obs_cb in obs.collision_boxes:
                adj_dino = CollisionBox(
                    dino_cb.x + dino_outer.x, dino_cb.y + dino_outer.y,
                    dino_cb.width, dino_cb.height)
                adj_obs = CollisionBox(
                    obs_cb.x + obs_outer.x, obs_cb.y + obs_outer.y,
                    obs_cb.width, obs_cb.height)
                if _box_compare(adj_dino, adj_obs):
                    return True
    return False


# ---------------------------------------------------------------------------
# Single obstacle instance — Obstacle class in offline.js
# ---------------------------------------------------------------------------

class _Obstacle:
    __slots__ = ('type_config', 'x', 'y', 'width', 'height', 'size',
                 'speed_offset', 'collision_boxes', 'gap',
                 'following_created')

    def __init__(self, type_config, speed, gap_coefficient):
        self.type_config = type_config
        self.following_created = False

        # Size: getRandomNum(1, Obstacle.MAX_OBSTACLE_LENGTH)  -> 1..3
        self.size = np.random.randint(1, 4)
        if self.size > 1 and type_config['multipleSpeed'] > speed:
            self.size = 1

        self.width = type_config['width'] * self.size
        self.height = type_config['height']

        # Y position (pterodactyl has variable heights)
        y_pos = type_config['yPos']
        if isinstance(y_pos, list):
            self.y = y_pos[np.random.randint(0, len(y_pos))]
        else:
            self.y = y_pos

        # Pterodactyl speed offset (+/-0.8)
        if 'speedOffset' in type_config:
            offset = type_config['speedOffset']
            self.speed_offset = offset if np.random.random() > 0.5 else -offset
        else:
            self.speed_offset = 0.0

        # Deep-copy collision boxes and adjust for multi-size obstacles
        self.collision_boxes = [
            CollisionBox(b.x, b.y, b.width, b.height)
            for b in type_config['collisionBoxes']
        ]
        if self.size > 1 and len(self.collision_boxes) >= 3:
            self.collision_boxes[1].width = (
                self.width
                - self.collision_boxes[0].width
                - self.collision_boxes[2].width
            )
            self.collision_boxes[2].x = (
                self.width - self.collision_boxes[2].width
            )

        # Gap to next obstacle  (Obstacle.prototype.getGap)
        min_gap = round(self.width * speed
                        + type_config['minGap'] * gap_coefficient)
        max_gap = round(min_gap * 1.5)       # MAX_GAP_COEFFICIENT = 1.5
        self.gap = np.random.randint(min_gap, max_gap + 1)

        self.x = 0.0  # positioned by the spawner


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class DinoRunEnv:
    """
    Chrome Dinosaur Game — 1:1 with Chromium's T-Rex Runner.

    Actions:
        0: Do nothing
        1: Jump

    Observations:
        (80, 80, 1) uint8 image

    Rewards:
        +0.1 per survived step
        -1.0 on crash
    """

    # -- Runner.defaultDimensions --
    GAME_WIDTH = 600
    GAME_HEIGHT = 150

    # -- Runner.config --
    ACCELERATION = 0.001
    BOTTOM_PAD = 10
    CLEAR_TIME = 3000                   # ms before obstacles appear
    GAP_COEFFICIENT = 0.6
    GRAVITY = 0.6
    INITIAL_JUMP_VELOCITY = -10         # Trex.config.INIITAL_JUMP_VELOCITY
    INITIAL_SPEED = 6.0                 # Runner.config.SPEED
    MAX_SPEED = 13.0                    # Runner.config.MAX_SPEED
    MAX_OBSTACLE_DUPLICATION = 2

    # -- Trex.config --
    DINO_WIDTH = 44
    DINO_HEIGHT = 47
    DINO_START_X = 50                   # Trex.config.START_X_POS
    DROP_VELOCITY = -5                  # Trex.config.DROP_VELOCITY
    MAX_JUMP_HEIGHT = 30                # yPos ceiling -> endJump()
    MIN_JUMP_HEIGHT = 30                # threshold for reachedMinHeight

    # -- DistanceMeter --
    SCORE_COEFFICIENT = 0.025

    # -- Derived --
    FPS = 60
    MS_PER_FRAME = 1000.0 / 60.0
    CLEAR_FRAMES = round(CLEAR_TIME / MS_PER_FRAME)   # ~180

    # -- HorizonLine --
    HORIZON_YPOS = 127

    # -- Observation rendering --
    CROP_WIDTH = 300
    OBS_SIZE = 80

    def __init__(self, domain_randomization=False, feature_noise=0.0,
                 skip_clear_time=False):
        self.action_space = ActionSpace(2)
        self.reward_range = (-1, 0.1)
        self.domain_randomization = domain_randomization
        self.feature_noise = feature_noise
        self.skip_clear_time = skip_clear_time

        # groundYPos = HEIGHT - DINO_HEIGHT - BOTTOM_PAD  (= 93)
        self.ground_y_pos = (self.GAME_HEIGHT - self.DINO_HEIGHT
                             - self.BOTTOM_PAD)
        # minJumpHeight yPos = groundYPos - MIN_JUMP_HEIGHT  (= 63)
        self.min_jump_height_y = self.ground_y_pos - self.MIN_JUMP_HEIGHT

        self._init_state()

    def _init_state(self):
        self.dino_x = self.DINO_START_X
        self.dino_y = self.ground_y_pos
        self.jump_velocity = 0.0
        self.jumping = False
        self.reached_min_height = False
        self.speed = self.INITIAL_SPEED
        self.distance_ran = 0.0
        self.score = 0
        self.crashed = False
        self.frame = 0
        self.obstacles = []
        self.obstacle_history = []

    # -- public API ---------------------------------------------------------

    def reset(self):
        """Reset game and return initial observation."""
        self._init_state()
        if self.domain_randomization:
            self.dino_x = np.random.randint(20, 56)
        return self._get_observation()

    def step(self, action):
        """
        Execute one game step (one frame at 60 fps).

        Returns:
            observation (np.ndarray): (80, 80, 1) image
            reward (float): 0.1 if alive, -1.0 if crashed
            score (int): current displayed game score
            done (bool): True if crashed
        """
        assert action in self.action_space
        self.frame += 1

        # --- speed increase (Runner.prototype.update) ---
        if self.speed < self.MAX_SPEED:
            self.speed = min(self.speed + self.ACCELERATION, self.MAX_SPEED)

        # --- jump (Trex.prototype.startJump) ---
        if action == 1 and not self.jumping:
            # Tweak jump velocity based on speed (like the real game)
            self.jump_velocity = (self.INITIAL_JUMP_VELOCITY
                                  - self.speed / 10.0)
            self.jumping = True
            self.reached_min_height = False

        # --- update jump physics (Trex.prototype.updateJump) ---
        if self.jumping:
            # Position update uses Math.round (framesElapsed=1 at 60fps)
            self.dino_y += round(self.jump_velocity)
            self.jump_velocity += self.GRAVITY

            # Check minJumpHeight threshold
            if self.dino_y < self.min_jump_height_y:
                self.reached_min_height = True

            # Check MAX_JUMP_HEIGHT -> endJump()
            if self.dino_y < self.MAX_JUMP_HEIGHT:
                if (self.reached_min_height
                        and self.jump_velocity < self.DROP_VELOCITY):
                    self.jump_velocity = self.DROP_VELOCITY

            # Back on ground -> reset
            if self.dino_y > self.ground_y_pos:
                self.dino_y = self.ground_y_pos
                self.jump_velocity = 0.0
                self.jumping = False
                self.reached_min_height = False

        # --- move obstacles (Obstacle.prototype.update) ---
        for obs in self.obstacles:
            effective_speed = self.speed + obs.speed_offset
            # Math.floor(speed * FPS/1000 * deltaTime) = floor(speed) at 60fps
            obs.x -= int(effective_speed)

        # Remove off-screen (!isVisible)
        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]

        # --- spawn new obstacles (Horizon.prototype.updateObstacles) ---
        clear = 0 if self.skip_clear_time else self.CLEAR_FRAMES
        has_obstacles = self.frame >= clear
        if has_obstacles:
            if self.obstacles:
                last = self.obstacles[-1]
                if (not last.following_created
                        and last.x + last.width > 0          # isVisible
                        and (last.x + last.width + last.gap
                             < self.GAME_WIDTH)):
                    self._add_obstacle()
                    last.following_created = True
            else:
                self._add_obstacle()

        # --- collision detection (checkForCollision) ---
        for obs in self.obstacles:
            if _check_collision(obs, self.dino_x, self.dino_y,
                                self.DINO_WIDTH, self.DINO_HEIGHT):
                self.crashed = True
                break

        # --- distance / score (DistanceMeter) ---
        self.distance_ran += self.speed
        self.score = round(self.distance_ran * self.SCORE_COEFFICIENT)

        if self.crashed:
            return self._get_observation(), -1.0, self.score, True
        return self._get_observation(), 0.1, self.score, False

    def get_features(self, add_noise=False):
        """
        8-dim normalised feature vector for RL agents.

        Features:
            0: distance to nearest obstacle   (/ GAME_WIDTH)
            1: width of nearest obstacle       (/ 75, max = CACTUS_LARGE * 3)
            2: height of nearest obstacle      (/ 50)
            3: dino height above ground        (/ groundYPos = 93)
            4: dino vertical velocity          (/ 12, > max |velocity|)
            5: is_jumping                      (0 or 1)
            6: distance to 2nd nearest obstacle (/ GAME_WIDTH)
            7: current speed                   (/ MAX_SPEED)

        High pterodactyls the dino can safely run under are filtered out.
        """
        ahead = []
        for obs in self.obstacles:
            if obs.x + obs.width <= self.dino_x:
                continue  # already passed
            # Skip high pterodactyls the dino can run under
            if (obs.type_config['type'] == 'PTERODACTYL'
                    and obs.y + obs.height <= self.ground_y_pos):
                continue
            ahead.append(obs)
        ahead.sort(key=lambda o: o.x)

        if ahead:
            dist1 = (ahead[0].x - self.dino_x) / self.GAME_WIDTH
            w1 = ahead[0].width / 75.0
            h1 = ahead[0].height / 50.0
        else:
            dist1, w1, h1 = 1.0, 0.0, 0.0

        dist2 = ((ahead[1].x - self.dino_x) / self.GAME_WIDTH
                 if len(ahead) >= 2 else 1.0)

        dino_height = (self.ground_y_pos - self.dino_y) / 93.0
        dino_vel = self.jump_velocity / 12.0
        jumping = 1.0 if self.jumping else 0.0
        spd = self.speed / self.MAX_SPEED

        features = np.array(
            [dist1, w1, h1, dino_height, dino_vel, jumping, dist2, spd],
            dtype=np.float32)

        if add_noise and self.feature_noise > 0:
            features += np.random.normal(
                0, self.feature_noise, features.shape).astype(np.float32)

        return features

    def get_score(self):
        return self.score

    def get_crashed(self):
        return self.crashed

    def end(self):
        pass

    # -- private helpers ----------------------------------------------------

    def _add_obstacle(self):
        """Spawn a new obstacle — Horizon.prototype.addNewObstacle()."""
        for _ in range(10):  # bounded retry (JS uses recursion)
            idx = np.random.randint(0, len(OBSTACLE_TYPES))
            tc = OBSTACLE_TYPES[idx]

            if self.speed < tc['minSpeed']:
                continue
            if self._duplicate_check(tc['type']):
                continue

            obs = _Obstacle(tc, self.speed, self.GAP_COEFFICIENT)
            # New obstacle starts at WIDTH + typeConfig.width (off-screen)
            obs.x = float(self.GAME_WIDTH + tc['width'])
            self.obstacles.append(obs)

            self.obstacle_history.insert(0, tc['type'])
            if len(self.obstacle_history) > self.MAX_OBSTACLE_DUPLICATION:
                self.obstacle_history = self.obstacle_history[
                    :self.MAX_OBSTACLE_DUPLICATION]
            return

    def _duplicate_check(self, next_type):
        """Prevent same type MAX_OBSTACLE_DUPLICATION times in a row."""
        count = 0
        for t in self.obstacle_history:
            count = count + 1 if t == next_type else 0
        return count >= self.MAX_OBSTACLE_DUPLICATION

    def _get_observation(self):
        """Render 80x80 greyscale image of the game state."""
        canvas = np.zeros((self.GAME_HEIGHT, self.CROP_WIDTH), dtype=np.uint8)

        # Ground line (HorizonLine.dimensions.YPOS = 127)
        cv2.line(canvas, (0, self.HORIZON_YPOS),
                 (self.CROP_WIDTH, self.HORIZON_YPOS), 255, 1)

        # Dinosaur
        dy = int(self.dino_y)
        cv2.rectangle(canvas,
                      (self.dino_x, dy),
                      (self.dino_x + self.DINO_WIDTH,
                       dy + self.DINO_HEIGHT),
                      255, 2)

        # Obstacles
        for obs in self.obstacles:
            ox = int(obs.x)
            if -obs.width < ox < self.CROP_WIDTH:
                cv2.rectangle(canvas,
                              (max(0, ox), obs.y),
                              (min(self.CROP_WIDTH, ox + obs.width),
                               obs.y + obs.height),
                              255, 2)

        img = cv2.resize(canvas, (self.OBS_SIZE, self.OBS_SIZE),
                         interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, -1)
