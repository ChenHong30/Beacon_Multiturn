#!/usr/bin/env python3
import json
import glob
import os
import argparse


def compute_strict_follow(input_path):
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        patterns = [os.path.join(input_path, "*.json"), os.path.join(input_path, "*.jsonl")]
        for p in patterns:
            files.extend(glob.glob(p))
    if not files:
        print(f"No log files found in {input_path}")
        return 1

    # counters per turn_id
    stats = {1: {"sum": 0, "n": 0}, 2: {"sum": 0, "n": 0}, 3: {"sum": 0, "n": 0}}
    total_results = 0

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            # try ndjson
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [json.loads(l) for l in f if l.strip()]
                # if it's ndjson, treat each line as a top-level object containing results
                data = {"results": []}
                for obj in lines:
                    if isinstance(obj, dict) and "results" in obj:
                        data["results"].extend(obj.get("results", []))
                    else:
                        # if each line is itself a result
                        if isinstance(obj, dict) and "turns" in obj:
                            data["results"].append(obj)
            except Exception:
                print(f"Failed to read {path}: {e}")
                continue

        results = data.get("results", [])
        for res in results:
            total_results += 1
            turns = res.get("turns", [])
            # build a map by turn_id for convenience
            turn_map = {t.get("turn_id"): t for t in turns if isinstance(t, dict) and "turn_id" in t}
            for tid in (1, 2, 3):
                t = turn_map.get(tid)
                if not t:
                    continue
                fs = t.get("follow_strict")
                # follow_strict expected to be a list of booleans (one per instruction)
                if fs is None:
                    continue
                try:
                    all_true = all(bool(x) for x in fs)
                except Exception:
                    # if it's a single boolean or unexpected type
                    all_true = bool(fs)
                stats[tid]["sum"] += 1 if all_true else 0
                stats[tid]["n"] += 1

    # print summary
    print(f"Processed {len(files)} files, total candidate results encountered: {total_results}")
    for tid in (1, 2, 3):
        s = stats[tid]["sum"]
        n = stats[tid]["n"]
        if n > 0:
            avg = s / n
            print(f"Turn {tid}: samples_with_turn={n}, all-strict-true_count={s}, mean(all-strict-true)={avg:.6f}")
        else:
            print(f"Turn {tid}: no samples with this turn found")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute per-turn strict-follow averages from a log JSON file or directory")
    p.add_argument("--logdir", default="logs", help="Directory containing log json files (default: logs)")
    p.add_argument("--file", help="Path to a single log file to process (overrides --logdir)")
    args = p.parse_args()
    path = args.file if args.file else args.logdir
    raise SystemExit(compute_strict_follow(path))
