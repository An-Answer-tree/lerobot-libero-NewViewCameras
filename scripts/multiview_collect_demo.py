import argparse
import os
from pathlib import Path

import init_path
from libero.libero import get_libero_path
from replay_dataset_utils import (
    DEFAULT_BENCHMARKS,
    discover_benchmark_tasks,
    reconstruct_dataset_file,
    validate_reconstructed_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay LIBERO source datasets to reconstruct semantically equivalent hdf5 files. "
            "Step3/4 (extra fixed cameras) are intentionally deferred."
        )
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Source dataset root, default get_libero_path('datasets').",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/szliutong/Desktop",
        help="Output dataset root, default '<source-root>_replay'.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to process. Default includes five LIBERO suites.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task allowlist by exact task names.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip target hdf5 file if already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target hdf5 file.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help=(
            "Camera names used during replay rendering. "
            "Default follows source env_args camera_names, fallback to robot0_eye_in_hand agentview."
        ),
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Replay render height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Replay render width.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Record depth observations during replay (off by default).",
    )
    parser.add_argument(
        "--no-proprio",
        action="store_true",
        help="Disable proprioceptive observation writing.",
    )
    parser.add_argument(
        "--state-error-threshold",
        type=float,
        default=0.01,
        help="Warn threshold for replay-vs-source state l2 divergence.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future parallelism. Current version always runs single-worker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved source/target mapping without reconstruction.",
    )
    return parser.parse_args()


def format_task(task_info):
    return (
        f"[{task_info['benchmark_name']}] {task_info['task_name']} "
        f"({task_info['relative_demo_path']})"
    )


def print_verify_commands(source_path, target_path):
    print("  verify commands:")
    print(f"    python scripts/get_dataset_info.py --dataset {source_path}")
    print(f"    python scripts/get_dataset_info.py --dataset {target_path}")


def main():
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite cannot be enabled together")

    if args.num_workers != 1:
        print("[warning] --num-workers is reserved; running with a single worker.")

    source_root = os.path.abspath(
        os.path.expanduser(args.source_root or get_libero_path("datasets"))
    )
    output_root = os.path.abspath(
        os.path.expanduser(args.output_root or f"{source_root.rstrip(os.sep)}_replay")
    )
    camera_names = args.camera_names if args.camera_names else None

    available_tasks, missing_tasks = discover_benchmark_tasks(
        source_root=source_root,
        benchmark_names=args.benchmarks,
        task_filter=args.tasks,
    )

    print(f"[info] source-root: {source_root}")
    print(f"[info] output-root: {output_root}")
    print(f"[info] selected benchmarks: {args.benchmarks}")
    print(f"[info] selected tasks: {'all' if args.tasks is None else len(args.tasks)}")
    print(f"[info] available source files: {len(available_tasks)}")
    print(f"[info] missing source files: {len(missing_tasks)}")

    for missing in missing_tasks:
        print(f"[warning] missing source demo: {missing['source_demo_path']}")

    if len(available_tasks) == 0:
        raise FileNotFoundError("No source hdf5 found for selected benchmark/task filters")

    if args.dry_run:
        for task_info in available_tasks:
            src = task_info["source_demo_path"]
            dst = os.path.join(output_root, task_info["relative_demo_path"])
            print(f"[dry-run] {format_task(task_info)}")
            print(f"          src={src}")
            print(f"          dst={dst}")
        return

    processed = 0
    skipped = 0
    failed = 0
    total_samples = 0

    for task_info in available_tasks:
        src_path = task_info["source_demo_path"]
        dst_path = os.path.join(output_root, task_info["relative_demo_path"])
        dst_file = Path(dst_path)
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():
            if args.skip_existing:
                skipped += 1
                print(f"[skip] {dst_path}")
                continue
            if args.overwrite:
                dst_file.unlink()
            else:
                raise FileExistsError(
                    f"Target file exists: {dst_path}. Use --skip-existing or --overwrite."
                )

        print(f"[replay] {format_task(task_info)}")
        try:
            summary = reconstruct_dataset_file(
                source_hdf5_path=src_path,
                output_hdf5_path=dst_path,
                camera_names=camera_names,
                use_depth=args.use_depth,
                no_proprio=args.no_proprio,
                divergence_threshold=args.state_error_threshold,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )
            validation = validate_reconstructed_file(src_path, dst_path)
        except Exception as exc:
            failed += 1
            print(f"[error] replay failed: {exc}")
            continue

        if not validation["ok"]:
            failed += 1
            print(f"[error] validation failed: {dst_path}")
            for error in validation["errors"]:
                print(f"  - {error}")
            print_verify_commands(src_path, dst_path)
            continue

        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"[warning] {warning}")

        processed += 1
        total_samples += summary["total_samples"]
        max_err = (
            max(ep["max_state_error"] for ep in summary["episodes"])
            if summary["episodes"]
            else 0.0
        )
        diverged = sum(
            ep["num_diverged_steps"] > 0 for ep in summary["episodes"]
        )
        print(
            "[ok] demos={num_demos}, transitions={num_samples}, max_state_error={max_err:.6f}, "
            "diverged_episodes={diverged}".format(
                num_demos=summary["num_demos"],
                num_samples=summary["total_samples"],
                max_err=max_err,
                diverged=diverged,
            )
        )
        print_verify_commands(src_path, dst_path)

    print("========================================")
    print(
        "[done] processed={processed}, skipped={skipped}, failed={failed}, "
        "total_transitions={total_samples}".format(
            processed=processed,
            skipped=skipped,
            failed=failed,
            total_samples=total_samples,
        )
    )
    print("[note] step3/step4 are deferred in this script.")

    if failed > 0:
        raise RuntimeError(f"Reconstruction finished with {failed} failed files")


if __name__ == "__main__":
    main()
