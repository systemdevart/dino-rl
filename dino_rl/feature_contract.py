"""
Shared feature contract for simulator training and browser inference.

The simulator and the browser player both expose the same 10-dimensional
feature vector. Keeping the normalization constants and filtering semantics in
one place makes drift much less likely.
"""

FEATURE_DIM = 10

# Normalization constants shared by env.get_features() and browser JS.
GAME_WIDTH = 600.0
GAME_HEIGHT = 150.0
GROUND_Y_POS = 93.0
MAX_OBSTACLE_WIDTH = 75.0
MAX_OBSTACLE_HEIGHT = 50.0
MAX_ABS_JUMP_VELOCITY = 12.0
MAX_GAME_SPEED = 13.0


def build_browser_state_js() -> str:
    """Return the browser-side JS that mirrors DinoRunEnv.get_features()."""
    return f"""
return (function() {{
    var r = Runner.instance_;
    if (!r || !r.tRex) return null;

    var tRex = r.tRex;
    var obstacles = r.horizon.obstacles;
    var canvasWidth = r.dimensions.WIDTH || {int(GAME_WIDTH)};
    var canvasHeight = r.dimensions.HEIGHT || {int(GAME_HEIGHT)};
    var groundY = tRex.groundYPos || {int(GROUND_Y_POS)};
    var ahead = [];

    for (var i = 0; i < obstacles.length; i++) {{
        var o = obstacles[i];
        var typeName = (o.typeConfig && o.typeConfig.type) || '';

        if (o.xPos + o.width <= tRex.xPos) {{
            continue;
        }}

        if (typeName === 'PTERODACTYL' &&
                o.yPos + o.typeConfig.height <= groundY) {{
            continue;
        }}

        ahead.push({{
            x: o.xPos,
            w: o.width,
            h: o.typeConfig.height,
            y: o.yPos
        }});
    }}

    ahead.sort(function(a, b) {{ return a.x - b.x; }});

    var dist1, w1, h1, y1;
    if (ahead.length >= 1) {{
        dist1 = (ahead[0].x - tRex.xPos) / canvasWidth;
        w1 = ahead[0].w / {MAX_OBSTACLE_WIDTH};
        h1 = ahead[0].h / {MAX_OBSTACLE_HEIGHT};
        y1 = ahead[0].y / canvasHeight;
    }} else {{
        dist1 = 1.0;
        w1 = 0.0;
        h1 = 0.0;
        y1 = 0.0;
    }}

    var dist2 = (ahead.length >= 2)
        ? (ahead[1].x - tRex.xPos) / canvasWidth
        : 1.0;

    var dinoHeight = Math.max(0, groundY - tRex.yPos) / {GROUND_Y_POS};
    var dinoVel = (tRex.jumpVelocity || 0.0) / {MAX_ABS_JUMP_VELOCITY};
    var jumping = tRex.jumping ? 1.0 : 0.0;
    var ducking = tRex.ducking ? 1.0 : 0.0;
    var speed = r.currentSpeed / {MAX_GAME_SPEED};
    var scoreStr = r.distanceMeter.digits.join('');

    return {{
        features: [
            dist1, w1, h1, dinoHeight, dinoVel,
            jumping, dist2, speed, y1, ducking
        ],
        crashed: r.crashed,
        score: parseInt(scoreStr, 10) || 0,
        playing: r.playing,
        distanceRan: r.distanceRan || 0.0
    }};
}})();
"""
