#!/usr/bin/env python
from __future__ import annotations

import argparse
import cProfile
import csv
import io
import pstats
import statistics
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
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV file for replicate timing summaries.",
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the terminal progress bar.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Run the same workload multiple times and summarize timing variance.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="Increment applied to the seed between replicates.",
    )
    return parser


def run_simulation(args: argparse.Namespace, *, seed: int) -> object:
    from src.project.model import GentrificationModel

    model = GentrificationModel(
        width=args.width,
        height=args.height,
        density=args.density,
        neighborhood_radius=args.neighborhood_radius,
        rng=seed,
        keep_game_history=not args.no_game_history,
        keep_agents=not args.no_agents,
    )

    for _ in range(args.warmup_steps):
        model.step()

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()

    progress_enabled = not args.no_progress
    progress_total = max(args.steps, 1)
    progress_width = 28
    progress_every = max(1, args.steps // 100)

    def render_progress(completed_steps: int) -> None:
        if not progress_enabled:
            return

        fraction = completed_steps / progress_total
        filled = min(progress_width, int(progress_width * fraction))
        bar = "#" * filled + "-" * (progress_width - filled)
        sys.stderr.write(
            f"\rProfiling [{bar}] {completed_steps}/{args.steps}"
        )
        sys.stderr.flush()

    render_progress(0)

    for step_index in range(args.steps):
        model.step()
        completed_steps = step_index + 1
        if progress_enabled and (
            completed_steps == args.steps
            or completed_steps % progress_every == 0
        ):
            render_progress(completed_steps)

    profiler.disable()
    elapsed = time.perf_counter() - start

    if progress_enabled:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return model, profiler, elapsed


def run_replicates(args: argparse.Namespace) -> list[dict[str, object]]:
    replicate_results: list[dict[str, object]] = []

    for replicate_index in range(args.replicates):
        seed = args.seed + replicate_index * args.seed_stride
        model, profiler, elapsed = run_simulation(args, seed=seed)
        replicate_results.append(
            {
                "replicate": replicate_index + 1,
                "seed": seed,
                "model": model,
                "profiler": profiler,
                "elapsed": elapsed,
            }
        )

    return replicate_results


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


def write_summary_csv(
    replicate_results: list[dict[str, object]],
    output_path: Path,
    *,
    steps: int,
    warmup_steps: int,
) -> None:
    """Write per-replicate timing results plus an aggregate summary."""
    elapsed_values = [float(result["elapsed"]) for result in replicate_results]
    per_step_values = [
        elapsed / max(steps, 1)
        for elapsed in elapsed_values
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "kind",
            "replicate",
            "seed",
            "steps",
            "warmup_steps",
            "population",
            "elapsed_seconds",
            "per_step_seconds",
        ])

        for result, elapsed, per_step in zip(
            replicate_results,
            elapsed_values,
            per_step_values,
            strict=True,
        ):
            writer.writerow([
                "replicate",
                result["replicate"],
                result["seed"],
                steps,
                warmup_steps,
                len(result["model"].agents),
                f"{elapsed:.6f}",
                f"{per_step:.6f}",
            ])

        writer.writerow([
            "summary",
            "",
            "",
            steps,
            warmup_steps,
            len(replicate_results[0]["model"].agents),
            f"{statistics.fmean(elapsed_values):.6f}",
            f"{statistics.fmean(per_step_values):.6f}",
        ])


def main() -> int:
    args = build_parser().parse_args()

    if args.replicates < 1:
        raise ValueError("replicates must be at least 1")

    replicate_results = run_replicates(args)

    elapsed_values = [float(result["elapsed"]) for result in replicate_results]
    per_step_values = [
        elapsed / max(args.steps, 1)
        for elapsed in elapsed_values
    ]

    print(
        "Run summary:",
        f"steps={args.steps}",
        f"warmup={args.warmup_steps}",
        f"replicates={args.replicates}",
        f"population={len(replicate_results[0]['model'].agents)}",
        f"mean_elapsed={statistics.fmean(elapsed_values):.3f}s",
        f"stdev_elapsed={(statistics.pstdev(elapsed_values) if len(elapsed_values) > 1 else 0.0):.3f}s",
        f"mean_per_step={statistics.fmean(per_step_values):.6f}s",
    )

    if args.replicates > 1:
        print("\nReplicate timings:")
        for result, elapsed in zip(replicate_results, elapsed_values, strict=True):
            print(
                f"  #{result['replicate']} seed={result['seed']} elapsed={elapsed:.3f}s per_step={elapsed / max(args.steps, 1):.6f}s"
            )

    profiler = replicate_results[0]["profiler"]
    model = replicate_results[0]["model"]
    elapsed = float(replicate_results[0]["elapsed"])

    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(args.profile_output)
    print(f"\nRaw profile written to: {args.profile_output}")

    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)

    print("\nTop profiling results (replicate 1):")
    write_report(
        profiler,
        sort_key=args.sort,
        top=args.top,
        text_output=args.text_output,
    )

    if args.text_output is not None:
        print(f"Text report written to: {args.text_output}")

    if args.summary_output is not None:
        write_summary_csv(
            replicate_results,
            args.summary_output,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        print(f"Summary CSV written to: {args.summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())