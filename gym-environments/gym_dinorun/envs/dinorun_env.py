"""
Pure Python Chrome Dinosaur Game environment.
Replaces the Selenium-based implementation with a fast simulation
that produces observations similar to Canny edge detection output.

Faithfully models the real game's acceleration: speed starts at 6 and
ramps up to 13 over ~7000 frames, with obstacle gaps scaling accordingly.
"""
import numpy as np
import cv2


class ActionSpace:
    """Minimal replacement for gym.spaces.Discrete."""
    def __init__(self, n):
        self.n = n

    def __contains__(self, x):
        return 0 <= x < self.n


class DinoRunEnv:
    """
    Chrome Dinosaur Game simulation with increasing speed.

    Actions:
        0: Do nothing
        1: Jump

    Observations:
        (80, 80, 1) uint8 image

    Rewards:
        +0.1 per survived step
        -1.0 on crash
    """

    # Game dimensions
    GAME_WIDTH = 600
    GAME_HEIGHT = 150
    GROUND_Y = 130

    # Dinosaur (matches real Chrome dino game physics)
    DINO_X = 50
    DINO_WIDTH = 44
    DINO_HEIGHT = 47
    DINO_HITBOX_MARGIN = 4
    JUMP_VELOCITY = -10.0
    GRAVITY = 0.6

    # Speed (matches real Chrome dino game)
    INITIAL_SPEED = 6.0
    ACCELERATION = 0.001
    MAX_SPEED = 13.0

    # Obstacle gaps (at initial speed; scale with speed)
    BASE_MIN_GAP = 300
    BASE_MAX_GAP = 500

    # Observation
    CROP_WIDTH = 300
    OBS_SIZE = 80

    def __init__(self):
        self.action_space = ActionSpace(2)
        self.reward_range = (-1, 0.1)
        self.dino_y = 0.0
        self.dino_vel_y = 0.0
        self.is_jumping = False
        self.score = 0
        self.distance = 0.0
        self.crashed = False
        self.obstacles = []
        self.speed = self.INITIAL_SPEED

    def reset(self):
        """Reset game and return initial observation."""
        self.dino_y = float(self.GROUND_Y - self.DINO_HEIGHT)
        self.dino_vel_y = 0.0
        self.is_jumping = False
        self.score = 0
        self.distance = 0.0
        self.crashed = False
        self.obstacles = []
        self.speed = self.INITIAL_SPEED
        # Spawn initial obstacle off-screen to the right
        x = self.GAME_WIDTH + np.random.randint(50, 150)
        self._spawn_obstacle(x)
        return self._get_observation()

    def _spawn_obstacle(self, x=None):
        """Spawn a new cactus obstacle with speed-scaled gap."""
        if x is None:
            if self.obstacles:
                last_x = max(obs[0] for obs in self.obstacles)
                # Scale gaps with speed so reaction time stays fair
                speed_factor = self.speed / self.INITIAL_SPEED
                min_gap = int(self.BASE_MIN_GAP * speed_factor)
                max_gap = int(self.BASE_MAX_GAP * speed_factor)
                gap = np.random.randint(min_gap, max_gap + 1)
                x = last_x + gap
            else:
                x = self.GAME_WIDTH + 50

        # Random obstacle size (small or large cactus)
        if np.random.random() < 0.5:
            width = np.random.randint(17, 35)
            height = 35
        else:
            width = np.random.randint(25, 51)
            height = 50

        self.obstacles.append([float(x), width, height])

    def step(self, action):
        """
        Execute one game step.

        Returns:
            observation (np.ndarray): (80, 80, 1) image
            reward (float): 0.1 if alive, -1.0 if crashed
            score (int): current game score
            done (bool): True if crashed
        """
        assert action in self.action_space

        # Accelerate (like the real game)
        if self.speed < self.MAX_SPEED:
            self.speed = min(self.speed + self.ACCELERATION, self.MAX_SPEED)

        # Jump (only if on ground)
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_vel_y = self.JUMP_VELOCITY

        # Update dino physics
        if self.is_jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y += self.GRAVITY
            ground_level = float(self.GROUND_Y - self.DINO_HEIGHT)
            if self.dino_y >= ground_level:
                self.dino_y = ground_level
                self.dino_vel_y = 0.0
                self.is_jumping = False

        # Move obstacles left at current speed
        for obs in self.obstacles:
            obs[0] -= self.speed

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs[0] + obs[1] > -10]

        # Spawn new obstacles when rightmost is on-screen
        rightmost = max((obs[0] for obs in self.obstacles), default=0)
        if rightmost < self.GAME_WIDTH:
            self._spawn_obstacle()

        # Check collision with forgiving hitbox
        m = self.DINO_HITBOX_MARGIN
        dino_left = self.DINO_X + m
        dino_top = int(self.dino_y) + m
        dino_right = self.DINO_X + self.DINO_WIDTH - m
        dino_bottom = int(self.dino_y) + self.DINO_HEIGHT - m

        for obs_x, obs_w, obs_h in self.obstacles:
            obs_left = int(obs_x)
            obs_top = self.GROUND_Y - obs_h
            obs_right = int(obs_x) + obs_w
            obs_bottom = self.GROUND_Y
            if (dino_right > obs_left and dino_left < obs_right and
                    dino_bottom > obs_top and dino_top < obs_bottom):
                self.crashed = True
                break

        # Update score (distance-based, like the real game)
        self.distance += self.speed
        self.score = int(self.distance) // 10

        if self.crashed:
            return self._get_observation(), -1.0, self.score, True
        return self._get_observation(), 0.1, self.score, False

    def _get_observation(self):
        """
        Render the game state as an 80x80 grayscale image.
        Cropped to focus on the area near and ahead of the dino.
        """
        canvas = np.zeros((self.GAME_HEIGHT, self.CROP_WIDTH), dtype=np.uint8)

        # Draw ground line
        cv2.line(canvas, (0, self.GROUND_Y), (self.CROP_WIDTH, self.GROUND_Y), 255, 1)

        # Draw dinosaur
        dy = int(self.dino_y)
        cv2.rectangle(canvas,
                      (self.DINO_X, dy),
                      (self.DINO_X + self.DINO_WIDTH, dy + self.DINO_HEIGHT),
                      255, 2)

        # Draw obstacles (only those within the crop)
        for obs_x, obs_w, obs_h in self.obstacles:
            ox = int(obs_x)
            if -obs_w < ox < self.CROP_WIDTH:
                cv2.rectangle(canvas,
                              (max(0, ox), self.GROUND_Y - obs_h),
                              (min(self.CROP_WIDTH, ox + obs_w), self.GROUND_Y),
                              255, 2)

        obs = cv2.resize(canvas, (self.OBS_SIZE, self.OBS_SIZE),
                         interpolation=cv2.INTER_AREA)
        obs = np.expand_dims(obs, -1)  # (80, 80, 1)
        return obs

    def get_features(self):
        """
        Return normalized feature vector describing the game state.

        Features (8 values):
            0: distance to nearest obstacle (normalized by GAME_WIDTH)
            1: width of nearest obstacle (normalized)
            2: height of nearest obstacle (normalized)
            3: dino y position (normalized, 0=ground, 1=max jump height)
            4: dino vertical velocity (normalized)
            5: is_jumping (0 or 1)
            6: distance to 2nd nearest obstacle (normalized)
            7: current game speed (normalized by MAX_SPEED)
        """
        ground_y = float(self.GROUND_Y - self.DINO_HEIGHT)

        # Sort obstacles by x position to find nearest
        ahead = sorted(
            [o for o in self.obstacles if o[0] + o[1] > self.DINO_X],
            key=lambda o: o[0]
        )

        if len(ahead) >= 1:
            dist1 = (ahead[0][0] - self.DINO_X) / self.GAME_WIDTH
            w1 = ahead[0][1] / 51.0
            h1 = ahead[0][2] / 50.0
        else:
            dist1, w1, h1 = 1.0, 0.0, 0.0

        if len(ahead) >= 2:
            dist2 = (ahead[1][0] - self.DINO_X) / self.GAME_WIDTH
        else:
            dist2 = 1.0

        dino_height = (ground_y - self.dino_y) / 90.0
        dino_vel = self.dino_vel_y / 10.0
        jumping = 1.0 if self.is_jumping else 0.0
        speed = self.speed / self.MAX_SPEED  # 0.46 at start, 1.0 at max

        return np.array([dist1, w1, h1, dino_height, dino_vel, jumping, dist2, speed],
                        dtype=np.float32)

    def get_score(self):
        return self.score

    def get_crashed(self):
        return self.crashed

    def end(self):
        pass
