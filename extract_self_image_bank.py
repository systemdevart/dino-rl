"""Extract a protected self-generated image replay anchor."""

import argparse
from pathlib import Path

import numpy as np
import torch

from image_dqn import SELF_REPLAY_ANCHOR_FORMAT


def chronological_replay_indices(size, capacity, write_index):
    """Return valid physical replay rows from oldest to newest."""
    if capacity < 1:
        raise ValueError("source replay capacity must be positive")
    if not 0 <= size <= capacity:
        raise ValueError("source replay size must be in [0, capacity]")
    if not 0 <= write_index < capacity:
        raise ValueError("source replay write index must be in [0, capacity)")
    if size < capacity:
        return np.arange(size, dtype=np.int64)
    return np.concatenate((
        np.arange(write_index, capacity, dtype=np.int64),
        np.arange(0, write_index, dtype=np.int64),
    ))


def terminal_replay_indices(dones, *, size, write_index):
    """Return valid physical replay rows containing terminal n-step returns."""
    dones = np.asarray(dones)
    if dones.ndim != 1:
        raise ValueError("source dones must be a 1-D array")

    capacity = len(dones)
    order = chronological_replay_indices(size, capacity, write_index)
    return order[dones[order] >= 0.5]


def select_anchor_indices(
        rewards, dones, *, size, write_index, target_size,
        recent_frac, recent_window, clear_reward_min, seed,
        terminal_frac=0.0, terminal_count=None):
    """Select deterministic, disjoint clear-reward and recent strata."""
    rewards = np.asarray(rewards)
    dones = np.asarray(dones)
    if rewards.ndim != 1 or dones.shape != rewards.shape:
        raise ValueError("source rewards and dones must be aligned 1-D arrays")
    capacity = len(rewards)
    order = chronological_replay_indices(size, capacity, write_index)
    if not 1 <= target_size <= size:
        raise ValueError("anchor size must be in [1, source replay size]")
    if not np.isfinite(recent_frac) or not 0.0 <= recent_frac <= 1.0:
        raise ValueError("recent_frac must be in [0, 1]")
    if recent_window < 1:
        raise ValueError("recent_window must be positive")
    if not np.isfinite(clear_reward_min):
        raise ValueError("clear_reward_min must be finite")
    if not np.isfinite(terminal_frac) or not 0.0 <= terminal_frac <= 1.0:
        raise ValueError("terminal_frac must be in [0, 1]")
    if terminal_count is not None:
        if terminal_count < 0:
            raise ValueError("terminal_count must be non-negative")
        if terminal_frac != 0.0:
            raise ValueError(
                "terminal_frac and terminal_count are mutually exclusive"
            )
        terminal_quota = int(terminal_count)
    else:
        terminal_quota = int(round(target_size * terminal_frac))
    if terminal_quota > target_size:
        raise ValueError("terminal count cannot exceed anchor size")

    rng = np.random.default_rng(seed)
    remaining_quota = target_size - terminal_quota
    recent_quota = int(round(remaining_quota * recent_frac))
    clear_quota = remaining_quota - recent_quota
    recent_pool = order[-min(recent_window, size):]
    clear_pool = order[
        (rewards[order] >= clear_reward_min) & (dones[order] < 0.5)
    ]

    selected_mask = np.zeros(capacity, dtype=bool)

    def choose(pool, count):
        count = min(int(count), len(pool))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(
            rng.choice(pool, size=count, replace=False), dtype=np.int64
        )

    if terminal_quota:
        terminal_pool = terminal_replay_indices(
            dones, size=size, write_index=write_index,
        )
        terminal_rows = choose(terminal_pool, terminal_quota)
        selected_mask[terminal_rows] = True
    else:
        terminal_pool = np.empty(0, dtype=np.int64)
        terminal_rows = np.empty(0, dtype=np.int64)

    clear_candidates = clear_pool[~selected_mask[clear_pool]]
    clear_rows = choose(clear_candidates, clear_quota)
    selected_mask[clear_rows] = True

    # If the clear stratum is sparse, spend its unused quota on recent rows.
    recent_candidates = recent_pool[~selected_mask[recent_pool]]
    recent_goal = recent_quota + (clear_quota - len(clear_rows))
    recent_rows = choose(recent_candidates, recent_goal)
    selected_mask[recent_rows] = True

    fill_count = (
        target_size - len(terminal_rows) - len(clear_rows) - len(recent_rows)
    )
    fill_candidates = order[~selected_mask[order]]
    fill_rows = choose(fill_candidates, fill_count)
    selected_mask[fill_rows] = True

    selected = np.concatenate((
        terminal_rows, clear_rows, recent_rows, fill_rows,
    ))
    if len(selected) != target_size or len(np.unique(selected)) != target_size:
        raise RuntimeError("selection policy failed to produce unique anchor rows")
    # Physical sorting makes mmap gathers substantially more sequential. Replay
    # sampling is random, so output row order has no training semantics.
    selected.sort()

    recent_mask = np.zeros(capacity, dtype=bool)
    recent_mask[recent_pool] = True
    stats = {
        "policy": (
            "terminal_clear_reward_and_recent_strata_v1"
            if terminal_quota else "clear_reward_and_recent_strata_v1"
        ),
        "target_size": int(target_size),
        "recent_frac": float(recent_frac),
        "recent_window": int(recent_window),
        "clear_reward_min": float(clear_reward_min),
        "clear_pool_size": int(len(clear_pool)),
        "recent_pool_size": int(len(recent_pool)),
        "assigned_clear_rows": int(len(clear_rows)),
        "assigned_recent_rows": int(len(recent_rows)),
        "assigned_fill_rows": int(len(fill_rows)),
        "selected_clear_rows": int(
            np.count_nonzero(
                (rewards[selected] >= clear_reward_min)
                & (dones[selected] < 0.5)
            )
        ),
        "selected_recent_rows": int(np.count_nonzero(recent_mask[selected])),
    }
    if terminal_quota:
        stats.update({
            "terminal_requested_rows": int(terminal_quota),
            "terminal_assigned_rows": int(len(terminal_rows)),
            "terminal_pool_size": int(len(terminal_pool)),
            "selected_terminal_rows": int(
                np.count_nonzero(dones[selected] >= 0.5)
            ),
        })
    return selected, stats


def validate_source_snapshot(payload):
    """Validate the live GPU replay fields needed by the extractor."""
    if not isinstance(payload, dict):
        raise ValueError("source snapshot must contain a dictionary")
    if payload.get("format") == SELF_REPLAY_ANCHOR_FORMAT:
        raise ValueError("source must be a live replay snapshot, not an anchor")
    fields = ("obs", "next_obs", "actions", "rewards", "dones", "disc")
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(f"source snapshot missing fields: {', '.join(missing)}")
    capacity = len(payload["actions"])
    if capacity < 1:
        raise ValueError("source replay capacity must be positive")
    for name in fields:
        if len(payload[name]) != capacity:
            raise ValueError(f"source field {name!r} has inconsistent capacity")
    size = int(payload.get("size", -1))
    write_index = int(payload.get("idx", -1))
    chronological_replay_indices(size, capacity, write_index)
    live_provenance = payload.get("live_replay_provenance")
    if (isinstance(live_provenance, dict)
            and live_provenance.get("teacher_or_demo_data") is True):
        raise ValueError("source provenance declares teacher/demo replay data")
    return size, capacity, write_index


def gather_rows(source, indices, dtype, chunk_size):
    """Gather mmap-backed rows without a full-size temporary tensor."""
    source = torch.as_tensor(source)
    output = torch.empty(
        (len(indices), *source.shape[1:]), dtype=dtype, device="cpu"
    )
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        rows = source.index_select(0, index_tensor[start:stop])
        output[start:stop].copy_(rows.to(dtype=dtype))
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a protected 30k TD-only anchor from a torch GPU replay "
            "snapshot. The source must be the learner's self-generated live "
            "replay, never a harvested teacher/demo bank."
        ),
        epilog=(
            "Example:\n"
            "  python extract_self_image_bank.py "
            "checkpoints/imgbuf_scratchfixn.pt "
            "checkpoints/imgbuf_scratchfixn_self30k.pt\n\n"
            "Then train with:\n"
            "  python image_dqn.py --self-bank "
            "checkpoints/imgbuf_scratchfixn_self30k.pt --self-frac 0.25 ...\n\n"
            "Selection is reproducible from the source snapshot, seed, and "
            "flags recorded in the output provenance. Do not pass an NPZ demo "
            "bank or a replay snapshot that was populated with teacher rows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="torch live GPU replay snapshot")
    parser.add_argument("output", help="output torch self-replay anchor")
    parser.add_argument("--size", type=int, default=30000,
                        help="anchor rows (default: 30000)")
    parser.add_argument("--recent-frac", type=float, default=0.5,
                        help="quota assigned to the recent stratum; the "
                             "remainder targets clear-reward rows (default: 0.5)")
    parser.add_argument("--recent-window", type=int, default=60000,
                        help="number of newest valid replay rows eligible for "
                             "the recent stratum (default: 60000)")
    parser.add_argument("--clear-reward-min", type=float, default=0.15,
                        help="minimum n-step reward treated as a clear-reward "
                             "row (default: 0.15)")
    terminal_quota = parser.add_mutually_exclusive_group()
    terminal_quota.add_argument(
        "--terminal-frac", type=float, default=0.0,
        help="fraction of anchor rows reserved for terminal n-step returns "
             "(default: 0, disabled)",
    )
    terminal_quota.add_argument(
        "--terminal-count", type=int,
        help="absolute anchor-row quota reserved for terminal n-step returns "
             "(default: disabled)",
    )
    parser.add_argument("--seed", type=int, default=20260713,
                        help="selection RNG seed (default: 20260713)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="rows gathered per mmap read (default: 512)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    source_path = Path(args.source)
    output_path = Path(args.output)
    payload = torch.load(
        source_path, map_location="cpu", weights_only=True, mmap=True
    )
    source_size, capacity, write_index = validate_source_snapshot(payload)
    rewards = torch.as_tensor(payload["rewards"]).numpy()
    dones = torch.as_tensor(payload["dones"]).numpy()
    selected, selection = select_anchor_indices(
        rewards, dones, size=source_size, write_index=write_index,
        target_size=args.size, recent_frac=args.recent_frac,
        recent_window=args.recent_window,
        clear_reward_min=args.clear_reward_min, seed=args.seed,
        terminal_frac=args.terminal_frac,
        terminal_count=args.terminal_count,
    )

    anchor = {
        "format": SELF_REPLAY_ANCHOR_FORMAT,
        "size": int(len(selected)),
        "capacity": int(len(selected)),
        "idx": 0,
        "obs": gather_rows(
            payload["obs"], selected, torch.uint8, args.chunk_size
        ),
        "next_obs": gather_rows(
            payload["next_obs"], selected, torch.uint8, args.chunk_size
        ),
        "actions": gather_rows(
            payload["actions"], selected, torch.long, args.chunk_size
        ),
        "rewards": gather_rows(
            payload["rewards"], selected, torch.float32, args.chunk_size
        ),
        "dones": gather_rows(
            payload["dones"], selected, torch.float32, args.chunk_size
        ),
        "disc": gather_rows(
            payload["disc"], selected, torch.float32, args.chunk_size
        ),
        "source_indices": torch.as_tensor(selected, dtype=torch.long),
        "provenance": {
            "kind": "self_generated_replay",
            "teacher_or_demo_data": False,
            "created_by": "extract_self_image_bank.py",
            "source": str(source_path.resolve()),
            "source_file_size": int(source_path.stat().st_size),
            "source_replay_size": int(source_size),
            "source_replay_capacity": int(capacity),
            "source_write_index": int(write_index),
            "seed": int(args.seed),
            "selection": selection,
            "source_live_replay_provenance": payload.get(
                "live_replay_provenance"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor, output_path)
    print(f"wrote {len(selected)}-row self replay anchor to {output_path}")
    print(f"provenance={anchor['provenance']!r}")


if __name__ == "__main__":
    main()
