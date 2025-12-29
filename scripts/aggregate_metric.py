import json
import glob
import os
from collections import defaultdict

def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0

def compute_multi_if_metrics(eval_results):
    total_samples = 0
    total_turns = 0
    total_instructions = 0

    strict_turn_pass = 0
    loose_turn_pass = 0

    strict_instruction_pass = 0
    loose_instruction_pass = 0

    strict_conv_pass = 0
    loose_conv_pass = 0

    # We don't need by_instruction for the user's requested output format, but good to have if needed later.
    # by_instruction = defaultdict(lambda: {"total": 0, "strict_pass": 0, "loose_pass": 0})

    for sample_res in eval_results:
        turns = sample_res.get("turns") or []
        # If a sample has no turns, usually we skip or count it as 0. 
        # The original script says: if not turns: continue
        if not turns:
            continue

        total_samples += 1
        conv_strict_ok = True
        conv_loose_ok = True

        for turn in turns:
            follow_strict = turn.get("follow_strict") or []
            follow_loose = turn.get("follow_loose") or []
            # instruction_ids = turn.get("instruction_id_list") or []

            total_turns += 1

            # Turn-level logic: All instructions in the turn must be passed
            strict_ok = bool(follow_strict) and all(bool(x) for x in follow_strict)
            loose_ok = bool(follow_loose) and all(bool(x) for x in follow_loose)

            strict_turn_pass += int(strict_ok)
            loose_turn_pass += int(loose_ok)

            conv_strict_ok = conv_strict_ok and strict_ok
            conv_loose_ok = conv_loose_ok and loose_ok

            total_instructions += len(follow_strict)
            strict_instruction_pass += sum(int(bool(x)) for x in follow_strict)
            loose_instruction_pass += sum(int(bool(x)) for x in follow_loose)

        strict_conv_pass += int(conv_strict_ok)
        loose_conv_pass += int(conv_loose_ok)

    return {
        "counts": {
            "samples": total_samples,
            "turns": total_turns,
            "instructions": total_instructions
        },
        "turn_level": {
            "strict_acc": _safe_div(strict_turn_pass, total_turns),
            "loose_acc": _safe_div(loose_turn_pass, total_turns)
        },
        "instruction_level": {
            "strict_acc": _safe_div(strict_instruction_pass, total_instructions),
            "loose_acc": _safe_div(loose_instruction_pass, total_instructions)
        },
        "conversation_level": {
            "strict_acc": _safe_div(strict_conv_pass, total_samples),
            "loose_acc": _safe_div(loose_conv_pass, total_samples)
        }
    }

def main():
    log_pattern = "/home/hkustgz/Beacon_Multiturn/logs/8_beacon_4_sink_recon_attention/multi_if_beacon_20251229_111547_8beacon_4sink.worker*.json"
    files = glob.glob(log_pattern)
    
    if not files:
        print(json.dumps({"error": f"No files found matching pattern: {log_pattern}"}))
        return

    all_eval_results = []
    
    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # The structure in the log file is {"results": [...], ...}
            results = data.get("results", [])
            all_eval_results.extend(results)
        except Exception as e:
            # Silently fail or log to stderr to keep stdout clean for JSON
            pass

    # Sort results just to be consistent (optional but good practice)
    # The original script sorts by 'global_index'
    all_eval_results.sort(key=lambda r: r.get("global_index", 0))

    metrics = compute_multi_if_metrics(all_eval_results)
    
    # Wrap in "metrics" key as requested
    output = {"metrics": metrics}
    
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()