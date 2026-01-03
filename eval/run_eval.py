#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TASK_SCRIPTS = {
    "gsm8k_interference": {
        "base": os.path.join(PROJECT_ROOT, "eval", "gsm8k_interference", "eval_gsm8k_interference_base.py"),
        "beacon": os.path.join(PROJECT_ROOT, "eval", "gsm8k_interference", "eval_gsm8k_interference_beacon.py"),
    },
    "mtbench_101": {
        "base": os.path.join(PROJECT_ROOT, "eval", "mtbench_101", "eval_mtbench_101_base.py"),
        "beacon": os.path.join(PROJECT_ROOT, "eval", "mtbench_101", "eval_mtbench_101_beacon.py"),
    },
    "multi_if": {
        "base": os.path.join(PROJECT_ROOT, "eval", "multi_if", "eval_multi_if_base.py"),
        "beacon": os.path.join(PROJECT_ROOT, "eval", "multi_if", "eval_multi_if_beacon.py"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluation tasks with a shared entrypoint.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_SCRIPTS.keys()),
        help="Evaluation task name.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["base", "beacon"],
        help="Model mode to run.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying task script. Prefix with -- to separate.",
    )

    args = parser.parse_args()

    script_path = TASK_SCRIPTS[args.task][args.mode]
    if not os.path.exists(script_path):
        raise SystemExit(f"Missing script: {script_path}")

    extra_args = list(args.extra_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    cmd = [sys.executable, script_path, *extra_args]
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
