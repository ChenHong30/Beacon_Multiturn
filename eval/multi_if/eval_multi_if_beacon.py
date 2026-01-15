import sys
import os
import re
import json
import multiprocessing as mp
import time
from datetime import datetime
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, set_seed

from tqdm.auto import tqdm

from metrics import eval_multi_if_sample
from modeling_qwen3 import Qwen3ForCausalLM as BeaconQwen3ForCausalLM


# ============================================================
# 1. Load Beacon model
# ============================================================
def load_beacon_model(
    model_path: str,
    device_id: int = 0,
    num_sinks: int = 1,
    num_beacons: int = 16,
):
    print(f"Loading Beacon model: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    config.num_sinks = num_sinks
    config.num_beacons_per_segment = num_beacons

    print(f"num_sinks = {num_sinks}")
    print(f"num_beacons_per_segment = {num_beacons}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model = BeaconQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        attn_implementation="eager",
    )

    model = model.to(device)
    model.eval()

    print(f"Beacon model loaded on {device}")
    return model, device


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _parse_cuda_ids(cuda_id: Any, cuda_ids: Optional[Any]) -> List[int]:
    """
    Parse GPU ids from either `cuda_ids` (preferred) or `cuda_id` (backward compatible).

    Notes:
      - `python-fire` may parse values like "0,1,2,3" into a tuple (0, 1, 2, 3).
      - We accept list/tuple/int/str for both params.
    """
    source = cuda_ids if cuda_ids is not None else cuda_id
    if source is None:
        return [0]

    if isinstance(source, (list, tuple)):
        ids = [int(x) for x in source]
    elif isinstance(source, str):
        s = source.strip()
        if not s:
            ids = [0]
        else:
            # accept e.g. "0,1,2,3" or "[0,1,2,3]"
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        ids = [int(x) for x in parsed]
                    else:
                        ids = [0]
                except Exception:
                    ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
            else:
                ids = [int(x) for x in re.split(r"[,\s]+", s) if x]
    else:
        ids = [int(source)]

    # de-dup but keep order
    ids = list(dict.fromkeys(ids))
    return ids


def _validate_cuda_ids(cuda_ids: List[int]) -> None:
    if not torch.cuda.is_available():
        if cuda_ids and cuda_ids != [0]:
            print(f"[WARN] CUDA not available; falling back to CPU (cuda_ids={cuda_ids}).")
        return

    device_count = torch.cuda.device_count()
    for cid in cuda_ids:
        if cid < 0 or cid >= device_count:
            raise ValueError(
                f"Invalid cuda_id={cid}; torch.cuda.device_count()={device_count}."
            )


def _get_english_indices_by_split(ds) -> Tuple[List[str], Dict[str, List[int]]]:
    split_names = list(ds.keys())
    english_indices_by_split: Dict[str, List[int]] = {}
    for split in split_names:
        langs = ds[split]["language"]
        english_indices_by_split[split] = [
            i for i, lang in enumerate(langs)
            if str(lang or "").lower() == "english"
        ]
    return split_names, english_indices_by_split


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"multi_if_beacon_{timestamp}_{run_tag}.worker{worker_id}.json"
    )


def _run_worker(
    *,
    worker_id: int,
    num_workers: int,
    cuda_id: int,
    model_path: str,
    num_sinks: int,
    num_beacons: int,
    log_dir: str,
    timestamp: str,
    run_tag: str,
    verbose: bool,
    flush_every: int,
    progress_counter: Optional[Any],
    task_queue: Any,
    seed: int = 42,
) -> None:
    set_seed(seed)
    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    # Initialize model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        model, device = load_beacon_model(
            model_path,
            device_id=cuda_id,
            num_sinks=num_sinks,
            num_beacons=num_beacons,
        )
        ds = load_dataset("facebook/Multi-IF")
    except Exception as e:
        print(f"[Worker {worker_id}] Init failed: {e}")
        return

    all_eval_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "num_sinks": num_sinks,
                "num_beacons_per_segment": num_beacons,
                "attempted": len(all_eval_results) + len(errors),
                "succeeded": len(all_eval_results),
                "failed": len(errors),
            },
            "results": all_eval_results,
            "errors": errors,
        }
        try:
            write_json_atomic(worker_out, output)
        except Exception as e:
            print(f"[WARN] Failed to write {worker_out}: {e}")

    flush()
    processed = 0

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            # Unpack task
            (split, en_idx, orig_idx, global_english_idx) = task

            try:
                sample = ds[split][orig_idx]
                generated_responses = run_multi_if_sample(
                    sample=sample,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    verbose=verbose,
                    num_beacons=num_beacons,
                )
                eval_result = eval_multi_if_sample(
                    sample=sample,
                    generated_responses=generated_responses,
                )
                eval_result["split"] = split
                eval_result["index"] = en_idx
                eval_result["orig_index"] = orig_idx
                eval_result["global_index"] = global_english_idx
                all_eval_results.append(eval_result)
            except Exception as e:
                errors.append(
                    {
                        "split": split,
                        "index": en_idx,
                        "orig_index": orig_idx,
                        "global_index": global_english_idx,
                        "key": sample.get("key") if 'sample' in locals() else "unknown",
                        "error": repr(e),
                    }
                )
            finally:
                processed += 1
                if progress_counter is not None:
                    try:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                    except Exception:
                        pass
                
                if flush_every > 0 and (processed % flush_every == 0):
                    flush()

    except Exception as e:
        print(f"[Worker {worker_id}] Crash: {e}")
    finally:
        flush()


# ============================================================
# 2. Encode dialogue
# ============================================================
def encode_dialogue(
    dialogue: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    device: torch.device
):
    text = tokenizer.apply_chat_template(
        dialogue,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


# ============================================================
# 3. Generate response
# ============================================================
def generate_stream(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens: int = 2048,
    num_beacons: int = 16,
):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            enable_beacon_compression=True,
            num_beacons=num_beacons,
        )

    input_len = input_ids.shape[1]
    new_tokens = output_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


# ============================================================
# 4. Extract turns from Multi-IF sample
# ============================================================
def extract_turns_from_multi_if(sample: Dict) -> List[Dict[str, str]]:
    turns = []

    pattern = re.compile(r"turn_(\d+)_prompt")
    indexed_prompts = []

    for k, v in sample.items():
        if v is None:
            continue
        m = pattern.fullmatch(k)
        if m:
            idx = int(m.group(1))
            indexed_prompts.append((idx, v))

    indexed_prompts.sort(key=lambda x: x[0])

    for idx, prompt_str in indexed_prompts:
        try:
            msg = json.loads(prompt_str)
            turns.append(msg)  # {"role": "user", "content": "..."}
        except json.JSONDecodeError as e:
            print(f"[WARN] turn_{idx}_prompt JSON decode failed: {e}")

    return turns


# ============================================================
# 5. Run one Multi-IF sample (ALL turns)
# ============================================================
def run_multi_if_sample(
    sample: Dict,
    model,
    tokenizer,
    device,
    *,
    verbose: bool = False,
    num_beacons: int = 16,
):
    dialogue = [
        {
            "role": "system",
            "content": "You are a helpful assistant, you should strictly follow every instruction given by the user."
        }
    ]

    turns = extract_turns_from_multi_if(sample)
    if verbose:
        print(f"Total turns: {len(turns)}")

    generated_responses: List[str] = []

    for turn_id, user_msg in enumerate(turns):
        if verbose:
            print("\n" + "-" * 60)
            print(f"Turn {turn_id + 1}")
            print(f"[User]\n{user_msg['content']}")

        dialogue.append(user_msg)

        input_ids, attention_mask = encode_dialogue(
            dialogue, tokenizer, device
        )

        if verbose:
            print(f"Context length: {input_ids.shape[1]} tokens")

        resp = generate_stream(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            num_beacons=num_beacons,
        )

        if verbose:
            print(f"[Assistant]\n{resp}")

        dialogue.append({"role": "assistant", "content": resp})
        generated_responses.append(resp)

    return generated_responses


# ============================================================
# 6. Metrics aggregation
# ============================================================
def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def compute_multi_if_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_samples = 0
    total_turns = 0
    total_instructions = 0

    strict_turn_pass = 0
    loose_turn_pass = 0

    strict_instruction_pass = 0
    loose_instruction_pass = 0

    strict_conv_pass = 0
    loose_conv_pass = 0

    by_instruction: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "strict_pass": 0, "loose_pass": 0}
    )

    for sample_res in eval_results:
        turns = sample_res.get("turns") or []
        if not turns:
            continue

        total_samples += 1
        conv_strict_ok = True
        conv_loose_ok = True

        for turn in turns:
            follow_strict = turn.get("follow_strict") or []
            follow_loose = turn.get("follow_loose") or []
            instruction_ids = turn.get("instruction_id_list") or []

            total_turns += 1

            strict_ok = bool(follow_strict) and all(bool(x) for x in follow_strict)
            loose_ok = bool(follow_loose) and all(bool(x) for x in follow_loose)

            strict_turn_pass += int(strict_ok)
            loose_turn_pass += int(loose_ok)

            conv_strict_ok = conv_strict_ok and strict_ok
            conv_loose_ok = conv_loose_ok and loose_ok

            total_instructions += len(follow_strict)
            strict_instruction_pass += sum(int(bool(x)) for x in follow_strict)
            loose_instruction_pass += sum(int(bool(x)) for x in follow_loose)

            for instruction_id, fs, fl in zip(instruction_ids, follow_strict, follow_loose):
                rec = by_instruction[str(instruction_id)]
                rec["total"] += 1
                rec["strict_pass"] += int(bool(fs))
                rec["loose_pass"] += int(bool(fl))

        strict_conv_pass += int(conv_strict_ok)
        loose_conv_pass += int(conv_loose_ok)

    by_instruction_acc = {
        k: {
            "total": v["total"],
            "strict_acc": _safe_div(v["strict_pass"], v["total"]),
            "loose_acc": _safe_div(v["loose_pass"], v["total"]),
        }
        for k, v in sorted(by_instruction.items(), key=lambda kv: kv[0])
    }

    return {
        "counts": {
            "samples": total_samples,
            "turns": total_turns,
            "instructions": total_instructions,
        },
        "turn_level": {
            "strict_acc": _safe_div(strict_turn_pass, total_turns),
            "loose_acc": _safe_div(loose_turn_pass, total_turns),
        },
        "instruction_level": {
            "strict_acc": _safe_div(strict_instruction_pass, total_instructions),
            "loose_acc": _safe_div(loose_instruction_pass, total_instructions),
        },
        "conversation_level": {
            "strict_acc": _safe_div(strict_conv_pass, total_samples),
            "loose_acc": _safe_div(loose_conv_pass, total_samples),
        },
        "by_instruction_id": by_instruction_acc,
    }


# ============================================================
# 7. Main: evaluate all English samples
# ============================================================
def main(
    model_path: str,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    num_sinks: int = 1,
    num_beacons: Optional[int] = None,
    flush_every: int = 1,
    num_workers: int = 16,
    seed: int = 42,
):
    set_seed(seed)
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if num_beacons is None:
        raise SystemExit("num_beacons is required. Pass --num_beacons=<int>.")
    num_beacons = int(num_beacons)
    if num_beacons < 1:
        raise SystemExit("num_beacons must be >= 1.")

    beacon_part = f"{int(num_beacons)}beacon"
    run_tag = f"{beacon_part}_{int(num_sinks)}sink"

    output_path = os.path.join(
        resolved_log_dir, f"multi_if_beacon_{timestamp}_{run_tag}.json"
    )

    ds = load_dataset("facebook/Multi-IF")

    cuda_id_list = _parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    _validate_cuda_ids(cuda_id_list)

    split_names, english_indices_by_split = _get_english_indices_by_split(ds)
    total_english = sum(len(v) for v in english_indices_by_split.values())
    
    # --------------------------------------------------------
    # Task Queue Setup
    # --------------------------------------------------------
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    
    # Fill Queue with tasks
    print(f"Generating tasks for {total_english} samples...")
    global_english_idx = 0
    for split in split_names:
        orig_indices = english_indices_by_split.get(split) or []
        for en_idx, orig_idx in enumerate(orig_indices):
            # Task tuple: (split, en_idx, orig_idx, global_english_idx)
            task_queue.put((split, en_idx, orig_idx, global_english_idx))
            global_english_idx += 1
            
    # Add termination signals
    final_num_workers = num_workers if (torch.cuda.is_available() and len(cuda_id_list) > 0) else 1
    
    # If no cuda devices or just 1 worker requested, we still use the queue logic.
    # Just need to ensure final_num_workers is logical.
    if not torch.cuda.is_available():
        final_num_workers = 1
        
    for _ in range(final_num_workers):
        task_queue.put(None)
        
    print(f"Launching {final_num_workers} workers on devices: {cuda_id_list or 'CPU'}...")
    
    worker_paths = [
        _worker_output_path(resolved_log_dir, timestamp, run_tag, wid)
        for wid in range(final_num_workers)
    ]

    progress_counter = ctx.Value("i", 0)
    procs: List[mp.Process] = []
    
    for wid in range(final_num_workers):
        # Round-robin GPU assignment
        if cuda_id_list:
            cid = cuda_id_list[wid % len(cuda_id_list)]
        else:
            cid = 0

        p = ctx.Process(
            target=_run_worker,
            kwargs={
                "worker_id": wid,
                "num_workers": final_num_workers,
                "cuda_id": cid,
                "model_path": model_path,
                "num_sinks": num_sinks,
                "num_beacons": num_beacons,
                "log_dir": resolved_log_dir,
                "timestamp": timestamp,
                "run_tag": run_tag,
                "verbose": verbose,
                "flush_every": flush_every,
                "progress_counter": progress_counter,
                "task_queue": task_queue,
            },
        )
        p.start()
        procs.append(p)

    # Monitor progress
    pbar = tqdm(total=total_english, desc="Evaluating Multi-IF (English)", unit="sample")
    last = 0
    try:
        while any(p.is_alive() for p in procs):
            try:
                with progress_counter.get_lock():
                    current = int(progress_counter.value)
            except Exception:
                current = last

            if current > last:
                step = min(current - last, total_english - last)
                if step > 0:
                    pbar.update(step)
                    last += step

            time.sleep(0.5)

        # Final update
        try:
            with progress_counter.get_lock():
                current = int(progress_counter.value)
        except Exception:
            current = last
        if current > last:
            step = min(current - last, total_english - last)
            if step > 0:
                pbar.update(step)
                last += step
    finally:
        pbar.close()

    for p in procs:
        p.join()

    # Collect results
    failed_workers = []
    for wid, p in enumerate(procs):
        if p.exitcode != 0:
            failed_workers.append(
                {
                    "worker_id": wid,
                    "exitcode": p.exitcode,
                }
            )

    all_eval_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    
    for wp in worker_paths:
        if not os.path.exists(wp):
            # If a worker didn't process anything (unlikely if N_tasks >> N_workers), it might not create a file?
            # Actually _run_worker calls flush() at start, so file should exist.
            print(f"[WARN] Missing worker output: {wp}")
            continue
        try:
            with open(wp, "r", encoding="utf-8") as f:
                d = json.load(f)
            all_eval_results.extend(d.get("results") or [])
            errors.extend(d.get("errors") or [])
        except Exception as e:
            print(f"[WARN] Failed to read {wp}: {e}")

    # Deduplication check? No, queue guarantees unique tasks.
    # Sorting by global index for consistency
    all_eval_results.sort(key=lambda r: r.get("global_index", 0))
    errors.sort(key=lambda r: r.get("global_index", 0))

    metrics = compute_multi_if_metrics(all_eval_results)
    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": final_num_workers,
            "num_splits": len(split_names),
            "splits": split_names,
            "total_english": total_english,
            "run_tag": run_tag,
            "num_beacons_per_segment": num_beacons,
            "num_sinks": num_sinks,
            "attempted": len(all_eval_results) + len(errors),
            "succeeded": len(all_eval_results),
            "failed": len(errors),
            "worker_outputs": worker_paths,
            "failed_workers": failed_workers,
        },
        "metrics": metrics,
        "results": all_eval_results,
        "errors": errors,
    }
    write_json_atomic(output_path, output)

    print("\n=== MULTI-IF (ENGLISH) METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")
    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{final_num_workers} workers failed; "
            f"partial results saved to: {output_path}"
        )
    return output_path


if __name__ == "__main__":
    try:
        import fire
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `fire`. Install with: pip install fire"
        ) from e

    fire.Fire(main)
