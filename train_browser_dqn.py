"""Train a feature-DQN FROM SCRATCH in the real browser (deterministic env).

With sim<->browser parity fixed (deterministic pause-and-step env, sim-parity
action semantics, case-insensitive bird filter), browser-native training uses
the same recipe that reaches ~6-7.5k in the sim: Dueling Double DQN with replay
and soft target updates (dino_rl.train_dqn.Agent), reward +0.01/frame alive,
-10 crash, +1 per obstacle cleared (detected feature-side: dist1 jumps up when
the nearest obstacle is passed). Decisions every frame (frames_per_action=1).

Usage:
    python train_browser_dqn.py [--num-envs 12] [--time-budget-sec 14400]
                                [--target 5000] [--out checkpoints/dino_browser_dqn.pth]
"""
import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import torch

from dino_rl.browser_env import ChromeDinoGame
from dino_rl.feature_contract import FEATURE_DIM
from dino_rl.networks import DuelingDQN
from dino_rl.train_dqn import Agent, greedy_feature_actions

CLEAR_JUMP = 0.25  # dist1 increase between consecutive alive frames => obstacle passed


def _maybe_apply_speed_curriculum(game, probability, speed_min, speed_max):
    if probability > 0.0 and np.random.rand() < probability:
        speed = float(np.random.uniform(speed_min, speed_max))
        game.driver.execute_script(
            "Runner.instance_.currentSpeed = arguments[0];", speed
        )


def _launch(start_speed_prob=0.0, start_speed_min=8.0, start_speed_max=13.0):
    g = ChromeDinoGame(headless=True)
    g.start()
    g.enable_deterministic()
    _maybe_apply_speed_curriculum(
        g, start_speed_prob, start_speed_min, start_speed_max
    )
    st = g.env_step(0, 1)
    return g, st


class NStepAccumulator:
    """Build discounted n-step rows independently for one browser worker."""

    def __init__(self, n, gamma):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def clear(self):
        self.buffer.clear()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        rows = []
        if done:
            while self.buffer:
                rows.append(self._emit())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n:
            rows.append(self._emit())
            self.buffer.popleft()
        return rows

    def _emit(self):
        total_reward = 0.0
        discount = 1.0
        last_next_state = self.buffer[0][3]
        last_done = False
        for _state, _action, reward, next_state, done in self.buffer:
            total_reward += discount * reward
            discount *= self.gamma
            last_next_state = next_state
            last_done = done
            if done:
                break
        state, action = self.buffer[0][:2]
        return (
            state,
            action,
            total_reward,
            last_next_state,
            last_done,
            discount,
        )


class VecFeatureBrowser:
    """N parallel deterministic browsers exposing the shared 10-dim features."""

    def __init__(
        self,
        n,
        start_speed_prob=0.0,
        start_speed_min=8.0,
        start_speed_max=13.0,
    ):
        self.n = n
        self.start_speed_prob = start_speed_prob
        self.start_speed_min = start_speed_min
        self.start_speed_max = start_speed_max
        self.pool = ThreadPoolExecutor(max_workers=n)
        res = list(self.pool.map(
            lambda _: _launch(
                start_speed_prob, start_speed_min, start_speed_max
            ),
            range(n),
        ))
        self.games = [r[0] for r in res]
        self.states = [r[1] for r in res]

    def get_feats(self):
        return np.array([s["features"] for s in self.states], dtype=np.float32)

    def _restart(self, game):
        game.restart_deterministic()
        _maybe_apply_speed_curriculum(
            game, self.start_speed_prob,
            self.start_speed_min, self.start_speed_max,
        )
        return game.env_step(0, 1)

    def step(self, actions, crash_penalty, alive_reward, clear_bonus):
        def one(i):
            g = self.games[i]
            prev_dist1 = float(self.states[i]["features"][0])
            try:
                st = g.env_step(int(actions[i]), 1)
            except Exception:
                st = None
            if st is None:
                # Infrastructure interruptions are not game crashes. Rebuild
                # the worker and tell the collector to discard its pending
                # n-step window rather than learning a synthetic -10 terminal.
                g.recover_session()
                g.start()
                g.enable_deterministic()
                _maybe_apply_speed_curriculum(
                    g, self.start_speed_prob,
                    self.start_speed_min, self.start_speed_max,
                )
                reset = g.env_step(0, 1)
                return dict(
                    sp=reset["features"], r=0.0, done=False, score=None,
                    reset=reset, recovered=True,
                )
            done = bool(st["crashed"])
            score = int(st["score"])
            if done:
                reset = self._restart(g)
                return dict(sp=st["features"], r=crash_penalty, done=True,
                            score=score, reset=reset, recovered=False)
            r = alive_reward
            if float(st["features"][0]) - prev_dist1 > CLEAR_JUMP:
                r += clear_bonus  # nearest obstacle switched => cleared one
            return dict(
                sp=st["features"], r=r, done=False, score=score,
                reset=st, recovered=False,
            )

        res = list(self.pool.map(one, range(self.n)))
        sp = np.array([r["sp"] for r in res], dtype=np.float32)
        rewards = np.array([r["r"] for r in res], dtype=np.float32)
        dones = np.array([r["done"] for r in res], dtype=bool)
        recoveries = np.array([r["recovered"] for r in res], dtype=bool)
        scores = [r["score"] for r in res if r["done"] and not r["recovered"]]
        for i, r in enumerate(res):
            self.states[i] = r["reset"]
        return sp, rewards, dones, scores, recoveries

    def close(self):
        for g in self.games:
            try:
                g.close()
            except Exception:
                pass
        self.pool.shutdown(wait=True)


def clone_feature_model_to_cpu(model):
    """Return an independent eval-mode CPU snapshot of a feature DQN."""
    action_size = model.advantage.out_features
    cpu_model = DuelingDQN(FEATURE_DIM, action_size).to("cpu")
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    cpu_model.load_state_dict(cpu_state)
    cpu_model.eval()
    return cpu_model


@torch.no_grad()
def _cpu_model_action(
    model,
    features,
    no_duck=False,
):
    state = torch.as_tensor(
        features, dtype=torch.float32, device="cpu"
    ).unsqueeze(0)
    q_values = model(state)
    return int(greedy_feature_actions(
        q_values,
        no_duck=no_duck,
    ).item())


def sample_random_feature_actions(
    features,
    *,
    no_duck=False,
    rng=np.random,
):
    """Sample exploratory actions from the configured action set."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (batch, {FEATURE_DIM})")
    action_count = 2 if no_duck else 3
    return rng.randint(0, action_count, size=len(features))


@torch.no_grad()
def evaluate(agent, episodes=3, cap=22000):
    if episodes <= 0:
        return []
    cpu_model = clone_feature_model_to_cpu(agent.model)
    inference_lock = Lock()

    def run_episode(_episode_index):
        game = ChromeDinoGame(headless=True)
        try:
            game.start()
            game.enable_deterministic()
            game.restart_deterministic()
            state = game.env_step(0, 1)
            steps = 0
            while steps < cap and state is not None and not state["crashed"]:
                with inference_lock:
                    action = _cpu_model_action(
                        cpu_model,
                        state["features"],
                        no_duck=agent.no_duck,
                    )
                state = game.env_step(action, 1)
                steps += 1
            return int(state["score"]) if state else 0
        finally:
            try:
                game.close()
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=min(episodes, 8)) as pool:
        return list(pool.map(run_episode, range(episodes)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--updates-per-tick", type=int, default=3,
                   help="replay updates per vec tick (~N/train_freq of the sim recipe)")
    p.add_argument("--n-step", type=int, default=1,
                   help="n-step return horizon; 1 preserves legacy behavior")
    p.add_argument("--terminal-frac", type=float, default=0.0,
                   help="minimum learner-batch fraction from cached terminal rows")
    p.add_argument("--sequence-replay-frac", type=float, default=0.0,
                   help="minimum learner-batch fraction from cached forensic sequence rows")
    p.add_argument("--speed-curriculum", type=float, default=0.0,
                   help="fraction of training episodes starting at speed U[min,max]")
    p.add_argument("--start-speed-min", type=float, default=8.0)
    p.add_argument("--start-speed-max", type=float, default=13.0)
    p.add_argument("--no-duck", action="store_true",
                   help="restrict policy to noop/jump and project raw duck Q below noop")
    p.add_argument("--no-duck-margin", type=float, default=0.01,
                   help="raw-checkpoint Q margin keeping duck below noop")
    p.add_argument("--empty-state-noop-margin", type=float, default=0.0,
                   help="grounded empty-state noop-over-jump margin; 0 disables")
    p.add_argument("--empty-state-noop-weight", type=float, default=0.0,
                   help="grounded empty-state hinge-loss weight; 0 disables")
    p.add_argument("--far-state-noop-distance", type=float, default=0.0,
                   help="minimum visible-obstacle distance in pixels; 0 disables")
    p.add_argument("--far-state-noop-margin", type=float, default=0.0,
                   help="grounded far-state noop-over-jump margin; 0 disables")
    p.add_argument("--far-state-noop-weight", type=float, default=0.0,
                   help="grounded far-state hinge-loss weight; 0 disables")
    p.add_argument("--sequence-jump-margin", type=float, default=0.0,
                   help="forensic sequence jump-over-noop margin; 0 disables")
    p.add_argument("--sequence-jump-weight", type=float, default=0.0,
                   help="forensic sequence hinge-loss weight; 0 disables")
    p.add_argument("--crash-penalty", type=float, default=-10.0)
    p.add_argument("--alive-reward", type=float, default=0.01)
    p.add_argument("--clear-bonus", type=float, default=1.0)
    p.add_argument("--eps-decay", type=float, default=0.995, help="per completed episode")
    p.add_argument("--eps-min", type=float, default=0.02)
    p.add_argument("--eval-every-sec", type=float, default=600)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--target", type=float, default=5000, help="early-stop eval avg")
    p.add_argument("--time-budget-sec", type=float, default=14400)
    p.add_argument("--out", default="checkpoints/dino_browser_dqn.pth")
    p.add_argument("--init-from", default="", help="checkpoint to continue from")
    p.add_argument("--min-replay", type=int, default=5000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--buffer-file", default="",
                   help="pickle path to persist the replay buffer across runs "
                        "(loaded at start if it exists, saved at each eval)")
    p.add_argument("--anneal-after", type=float, default=1500,
                   help="once best eval exceeds this, decay lr 0.6x per new best "
                        "(floor 1e-5) to stop TD churn from flipping a good policy")
    args = p.parse_args()
    if args.n_step < 1:
        p.error("--n-step must be at least 1")
    if not 0.0 <= args.terminal_frac <= 1.0:
        p.error("--terminal-frac must be in [0, 1]")
    if not 0.0 <= args.sequence_replay_frac <= 1.0:
        p.error("--sequence-replay-frac must be in [0, 1]")
    if args.terminal_frac + args.sequence_replay_frac > 1.0:
        p.error("--terminal-frac + --sequence-replay-frac must be at most 1")
    if not 0.0 <= args.speed_curriculum <= 1.0:
        p.error("--speed-curriculum must be in [0, 1]")
    if args.start_speed_min > args.start_speed_max:
        p.error("--start-speed-min cannot exceed --start-speed-max")
    if (
        not np.isfinite(args.empty_state_noop_margin)
        or args.empty_state_noop_margin < 0
    ):
        p.error("--empty-state-noop-margin must be finite and non-negative")
    if (
        not np.isfinite(args.empty_state_noop_weight)
        or args.empty_state_noop_weight < 0
    ):
        p.error("--empty-state-noop-weight must be finite and non-negative")

    if (
        not np.isfinite(args.far_state_noop_distance)
        or args.far_state_noop_distance < 0
    ):
        p.error("--far-state-noop-distance must be finite and non-negative")
    if (
        not np.isfinite(args.far_state_noop_margin)
        or args.far_state_noop_margin < 0
    ):
        p.error("--far-state-noop-margin must be finite and non-negative")
    if (
        not np.isfinite(args.far_state_noop_weight)
        or args.far_state_noop_weight < 0
    ):
        p.error("--far-state-noop-weight must be finite and non-negative")
    if (
        not np.isfinite(args.sequence_jump_margin)
        or args.sequence_jump_margin < 0
    ):
        p.error("--sequence-jump-margin must be finite and non-negative")
    if (
        not np.isfinite(args.sequence_jump_weight)
        or args.sequence_jump_weight < 0
    ):
        p.error("--sequence-jump-weight must be finite and non-negative")

    agent = Agent(
        3,
        no_duck=args.no_duck,
        no_duck_margin=args.no_duck_margin,
        empty_state_noop_margin=args.empty_state_noop_margin,
        empty_state_noop_weight=args.empty_state_noop_weight,
        far_state_noop_distance=args.far_state_noop_distance,
        far_state_noop_margin=args.far_state_noop_margin,
        far_state_noop_weight=args.far_state_noop_weight,
        sequence_jump_margin=args.sequence_jump_margin,
        sequence_jump_weight=args.sequence_jump_weight,
    )
    agent.weight_backup = args.out
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=True)
        agent.model.load_state_dict(ck["model"])
        agent.target_model.load_state_dict(ck.get("target_model", ck["model"]))
        agent.project_no_duck()
        print(f"continued from {args.init_from}", flush=True)
    agent.epsilon = args.eps_start
    agent.epsilon_min = args.eps_min
    agent.epsilon_decay = args.eps_decay
    agent.min_replay_size = args.min_replay
    if args.buffer_file:
        import os
        import pickle
        if os.path.isfile(args.buffer_file):
            with open(args.buffer_file, "rb") as fh:
                saved_replay = pickle.load(fh)
            for tr in saved_replay:
                agent.remember(*tr)
            print(f"loaded replay buffer: {len(agent.memory)} transitions "
                  f"from {args.buffer_file}"
                  + (f" ({len(saved_replay) - len(agent.memory)} illegal "
                     "action-2 rows skipped)"
                     if args.no_duck else ""),
                  flush=True)
    device = next(agent.model.parameters()).device
    print(
        f"browser DQN from scratch | envs={args.num_envs} fpa=1 "
        f"device={device} eval_device=cpu clear_bonus={args.clear_bonus} "
        f"n_step={args.n_step} terminal_frac={args.terminal_frac} "
        f"sequence_replay_frac={args.sequence_replay_frac} "
        f"speed_curriculum={args.speed_curriculum} no_duck={args.no_duck} "
        f"empty_state_noop_margin={args.empty_state_noop_margin} "
        f"empty_state_noop_weight={args.empty_state_noop_weight} "
        f"far_state_noop_distance={args.far_state_noop_distance} "
        f"far_state_noop_margin={args.far_state_noop_margin} "
        f"far_state_noop_weight={args.far_state_noop_weight} "
        f"sequence_jump_margin={args.sequence_jump_margin} "
        f"sequence_jump_weight={args.sequence_jump_weight}",
        flush=True,
    )

    vec = VecFeatureBrowser(
        args.num_envs,
        start_speed_prob=args.speed_curriculum,
        start_speed_min=args.start_speed_min,
        start_speed_max=args.start_speed_max,
    )
    N = args.num_envs
    nstep_accumulators = [
        NStepAccumulator(args.n_step, agent.gamma) for _ in range(N)
    ]
    best_eval = -1.0
    best_min = -1
    env_steps = 0
    episodes_done = 0
    recent = []
    start = time.time()
    last_eval = start

    while time.time() - start < args.time_budget_sec:
        feats = vec.get_feats()
        # batched epsilon-greedy
        with torch.no_grad():
            feats_t = torch.as_tensor(
                feats, dtype=torch.float32, device=device
            )
            q = agent.model(feats_t)
            greedy = agent.greedy_actions(q).cpu().numpy()
        explore = np.random.rand(N) < agent.epsilon
        random_actions = sample_random_feature_actions(
            feats,
            no_duck=args.no_duck,
        )
        actions = np.where(
            explore,
            random_actions,
            greedy,
        )

        sp, rewards, dones, scores, recoveries = vec.step(
            actions, args.crash_penalty, args.alive_reward, args.clear_bonus)
        for i in range(N):
            if recoveries[i]:
                nstep_accumulators[i].clear()
                continue
            if args.n_step == 1:
                agent.remember(
                    feats[i], int(actions[i]), float(rewards[i]),
                    sp[i], bool(dones[i]),
                )
            else:
                for transition in nstep_accumulators[i].push(
                    feats[i], int(actions[i]), float(rewards[i]),
                    sp[i], bool(dones[i]),
                ):
                    agent.remember(*transition)
        env_steps += N
        for _ in scores:
            episodes_done += 1
            agent.decay_epsilon()
        recent.extend(scores)

        if len(agent.memory) >= agent.min_replay_size:
            for _ in range(args.updates_per_tick):
                agent.replay(
                    args.batch_size,
                    terminal_frac=args.terminal_frac,
                    sequence_frac=args.sequence_replay_frac,
                )
                agent.soft_update_target()

        if time.time() - last_eval >= args.eval_every_sec and \
                len(agent.memory) >= agent.min_replay_size:
            sps = env_steps / (time.time() - start)
            tr = np.mean(recent[-50:]) if recent else 0.0
            print(f"[{int(time.time()-start)}s] eps={agent.epsilon:.3f} "
                  f"episodes={episodes_done} env_steps={env_steps} steps/s={sps:.0f} "
                  f"train_score~{tr:.0f}", flush=True)
            ev = evaluate(agent, args.eval_episodes)
            avg = float(np.mean(ev))
            min_score = int(np.min(ev))
            print(f"  >> EVAL avg={avg:.0f} min={min_score} scores={ev}",
                  flush=True)
            if (min_score, avg) > (best_min, best_eval):
                best_min, best_eval = min_score, avg
                agent.save_model(args.out)
                print(f"  >> saved best (min={best_min} avg={best_eval:.0f}) "
                      f"-> {args.out}", flush=True)
                if best_min > args.anneal_after:
                    for g in agent.optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.6, 1e-5)
                    print(f"  >> lr annealed to {agent.optimizer.param_groups[0]['lr']:.1e}",
                          flush=True)
            else:
                agent.save_model(args.out + ".last")
            if args.buffer_file:
                import pickle
                with open(args.buffer_file + ".tmp", "wb") as fh:
                    pickle.dump(list(agent.memory), fh, protocol=4)
                import os
                os.replace(args.buffer_file + ".tmp", args.buffer_file)
            if min_score >= args.target:
                print(f"TARGET REACHED: eval min {min_score} >= {args.target}",
                      flush=True)
                break
            last_eval = time.time()

    print(f"DONE episodes={episodes_done} env_steps={env_steps} "
          f"best_min={best_min} best_eval={best_eval:.0f}", flush=True)
    vec.close()


if __name__ == "__main__":
    main()
