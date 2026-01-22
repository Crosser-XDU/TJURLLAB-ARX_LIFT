import argparse
import sys
from typing import Iterable, Iterator, List, Optional

import numpy as np

from gym_robot_env4arx import ARXsingleArmRobotEnv


def parse_action_line(line: str) -> Optional[np.ndarray]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = [p for p in stripped.replace(",", " ").split(" ") if p]
    if len(parts) != 7:
        raise ValueError(f"Expected 7 values per action, got {len(parts)}: {line!r}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def iter_actions(lines: Iterable[str]) -> Iterator[np.ndarray]:
    for line in lines:
        action = parse_action_line(line)
        if action is not None:
            yield action


def iter_chunks(actions: Iterable[np.ndarray], chunk_size: int) -> Iterator[List[np.ndarray]]:
    chunk: List[np.ndarray] = []
    for action in actions:
        chunk.append(action)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        print("Warning: Incomplete final chunk, executing anyway.", file=sys.stderr)
        yield chunk

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ARX env by executing actions in fixed-size chunks."
    )
    parser.add_argument(
        "--control-type",
        choices=["end", "joint"],
        default="end",
        help="Control space to use.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["abs", "delta"],
        default="abs",
        help="Interpret actions as absolute or delta targets.",
    )
    parser.add_argument(
        "--camera-type",
        choices=["color", "depth", "all"],
        default="all",
        help="Camera stream type.",
    )
    parser.add_argument("--freq", type=int, default=20, help="Control frequency (Hz).")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of actions to execute per chunk.",
    )
    parser.add_argument(
        "--actions-file",
        default="action4test.txt",
        help="Path to action list file; '-' reads from stdin.",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory to save camera images.",
    )
    args = parser.parse_args()

    env = ARXsingleArmRobotEnv(
        control_type=args.control_type,
        control_mode=args.control_mode,
        camera_type=args.camera_type,
        freq=args.freq,
        dir=args.dir,
    )
    try:
        env.reset()
        if args.actions_file == "-":
            lines = sys.stdin
        else:
            with open(args.actions_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        actions = iter_actions(lines)
        for chunk in iter_chunks(actions, args.chunk_size):
            for action in chunk:
                _, pic, _, _ = env.step(action)

                
    finally:
        env.close()


if __name__ == "__main__":
    main()
