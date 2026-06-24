#!/usr/bin/env python
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the GentrificationModel with cProfile and print the "
            "slowest functions."
        )
    )
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("profile_model.prof"),
        help="Write the raw cProfile data to this file.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional text report path. If omitted, the report prints to stdout.",
    )
    parser.add_argument(
        "--sort",
        choices=("cumulative", "tottime", "calls"),
        default="cumulative",
        help="Sort key used for the printed function table.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many rows to print from the profiling table.",
    )
    parser.add_argument(
        "--no-game-history",
        action="store_true",
        help="Disable game history storage while profiling.",
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable per-agent history storage while profiling.",
    )
    return parser


def run_simulation(args: argparse.Namespace) -> object:
    from src.project.model import GentrificationModel

    model = GentrificationModel(
        width=args.width,
        height=args.height,
        density=args.density,
        neighborhood_radius=args.neighborhood_radius,
        rng=args.seed,
        keep_game_history=not args.no_game_history,
        keep_agents=not args.no_agents,
    )

    for _ in range(args.warmup_steps):
        model.step()

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()

    for _ in range(args.steps):
        model.step()

    profiler.disable()
    elapsed = time.perf_counter() - start

    return model, profiler, elapsed


def write_report(
    profiler: cProfile.Profile,
    *,
    sort_key: str,
    top: int,
    text_output: Path | None,
) -> None:
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.strip_dirs().sort_stats(sort_key).print_stats(top)

    report = buffer.getvalue()
    print(report, end="")

    if text_output is not None:
        with text_output.open("w", encoding="utf-8") as handle:
            handle.write(report)


def main() -> int:
    args = build_parser().parse_args()

    model, profiler, elapsed = run_simulation(args)

    print(
        "Run summary:",
        f"steps={args.steps}",
        f"warmup={args.warmup_steps}",
        f"population={len(model.agents)}",
        f"elapsed={elapsed:.3f}s",
        f"per_step={elapsed / max(args.steps, 1):.6f}s",
    )

    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(args.profile_output)
    print(f"Raw profile written to: {args.profile_output}")

    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)

    print("\nTop profiling results:")
    write_report(
        profiler,
        sort_key=args.sort,
        top=args.top,
        text_output=args.text_output,
    )

    if args.text_output is not None:
        print(f"Text report written to: {args.text_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())