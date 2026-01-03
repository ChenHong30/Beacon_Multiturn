import sys
import os
import json
import multiprocessing as mp
import time
from datetime import datetime
from typing import Any, List, Dict, Optional

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from tqdm.auto import tqdm

from metrics import eval_multi_if_sample

from eval import common, modeling
from eval.multi_if import multi_if_utils


# ============================================================
# 1. Worker Helpers
# ============================================================
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
    log_dir: str,
    timestamp: str,
    run_tag: str,
    split_names: List[str],
    english_indices_by_split: Dict[str, List[int]],
    verbose: bool,
    flush_every: int,
    progress_counter: Optional[Any],
) -> None:
    if num_workers > 1:
        common.configure_process_logging(rank=worker_id, force_non_main=True)

    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model, device = modeling.load_beacon_model(
        model_path,
        device_id=cuda_id,
        num_sinks=num_sinks,
        strict_num_beacons=True,
    )

    ds = load_dataset("facebook/Multi-IF")

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
                "splits": split_names,
                "attempted": len(all_eval_results) + len(errors),
                "succeeded": len(all_eval_results),
                "failed": len(errors),
            },
            "results": all_eval_results,
            "errors": errors,
        }
        try:
            common.write_json_atomic(worker_out, output)
        except Exception as e:
            print(f"[WARN] Failed to write {worker_out}: {e}")

    flush()
    try:
        global_english_idx = 0
        processed = 0
        for split in split_names:
            orig_indices = english_indices_by_split.get(split) or []
            for en_idx, orig_idx in enumerate(orig_indices):
                if global_english_idx % num_workers != worker_id:
                    global_english_idx += 1
                    continue

                sample = ds[split][orig_idx]
                try:
                    generated_responses = run_multi_if_sample(
                        sample=sample,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        verbose=verbose,
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
                            "key": sample.get("key"),
                            "error": repr(e),
                        }
                    )
                finally:
                    global_english_idx += 1
                    processed += 1
                    if progress_counter is not None:
                        try:
                            with progress_counter.get_lock():
                                progress_counter.value += 1
                        except Exception:
                            pass
                    if flush_every > 0 and (processed % flush_every == 0):
                        flush()
    finally:
        pass

    flush()


# ============================================================
# 2. Run one Multi-IF sample (ALL turns)
# ============================================================
def run_multi_if_sample(
    sample: Dict,
    model,
    tokenizer,
    device,
    *,
    verbose: bool = False,
):
    dialogue = [
        {
            "role": "system",
            "content": "You are a helpful assistant, you should strictly follow every instruction given by the user."
        }
    ]

    turns = multi_if_utils.extract_turns_from_multi_if(sample)
    if verbose:
        print(f"Total turns: {len(turns)}")

    generated_responses: List[str] = []

    for turn_id, user_msg in enumerate(turns):
        if verbose:
            print("\n" + "-" * 60)
            print(f"Turn {turn_id + 1}")
            print(f"[User]\n{user_msg['content']}")

        dialogue.append(user_msg)

        prompt_text = tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_ids, attention_mask = modeling.encode_prompt(tokenizer, prompt_text, device)

        if verbose:
            print(f"Context length: {input_ids.shape[1]} tokens")

        resp = modeling.generate_beacon_response(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            top_p=None,
            apply_temperature_if_sampled=False,
        )

        if verbose:
            print(f"[Assistant]\n{resp}")

        dialogue.append({"role": "assistant", "content": resp})
        generated_responses.append(resp)

    return generated_responses


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
    flush_every: int = 1,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_beacons_per_segment: Optional[int] = None
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_beacons_per_segment = getattr(cfg, "num_beacons_per_segment", None)
    except Exception as e:
        print(f"[WARN] Failed to read model config from {model_path}: {e}")

    beacon_part = (
        f"{int(num_beacons_per_segment)}beacon"
        if num_beacons_per_segment is not None
        else "unknownbeacon"
    )
    run_tag = f"{beacon_part}_{int(num_sinks)}sink"

    output_path = os.path.join(
        resolved_log_dir, f"multi_if_beacon_{timestamp}_{run_tag}.json"
    )

    ds = load_dataset("facebook/Multi-IF")

    cuda_id_list = common.parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    common.validate_cuda_ids(cuda_id_list)

    split_names, english_indices_by_split = multi_if_utils.get_english_indices_by_split(ds)
    total_english = sum(len(v) for v in english_indices_by_split.values())

    # ============================================================
    # Multi-GPU: one process per GPU, shard by global_index % num_workers
    # ============================================================
    if torch.cuda.is_available() and len(cuda_id_list) > 1:
        num_workers = len(cuda_id_list)
        worker_paths = [
            _worker_output_path(resolved_log_dir, timestamp, run_tag, wid)
            for wid in range(num_workers)
        ]

        ctx = mp.get_context("spawn")
        progress_counter = ctx.Value("i", 0)
        procs: List[mp.Process] = []
        for wid, cid in enumerate(cuda_id_list):
            p = ctx.Process(
                target=_run_worker,
                kwargs={
                    "worker_id": wid,
                    "num_workers": num_workers,
                    "cuda_id": cid,
                    "model_path": model_path,
                    "num_sinks": num_sinks,
                    "log_dir": resolved_log_dir,
                    "timestamp": timestamp,
                    "run_tag": run_tag,
                    "split_names": split_names,
                    "english_indices_by_split": english_indices_by_split,
                    "verbose": verbose,
                    "flush_every": flush_every,
                    "progress_counter": progress_counter,
                },
            )
            p.start()
            procs.append(p)

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

                time.sleep(0.2)

            # Final update after all workers finished.
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

        failed_workers = []
        for wid, p in enumerate(procs):
            if p.exitcode != 0:
                failed_workers.append(
                    {
                        "worker_id": wid,
                        "cuda_id": cuda_id_list[wid],
                        "exitcode": p.exitcode,
                    }
                )

        all_eval_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for wp in worker_paths:
            if not os.path.exists(wp):
                print(f"[WARN] Missing worker output: {wp}")
                continue
            try:
                with open(wp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                all_eval_results.extend(d.get("results") or [])
                errors.extend(d.get("errors") or [])
            except Exception as e:
                print(f"[WARN] Failed to read {wp}: {e}")

        all_eval_results.sort(key=lambda r: r.get("global_index", 0))
        errors.sort(key=lambda r: r.get("global_index", 0))

        metrics = multi_if_utils.compute_multi_if_metrics(all_eval_results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_ids": cuda_id_list,
                "log_dir": resolved_log_dir,
                "num_workers": num_workers,
                "num_splits": len(split_names),
                "splits": split_names,
                "total_english": total_english,
                "run_tag": run_tag,
                "num_beacons_per_segment": num_beacons_per_segment,
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
        common.write_json_atomic(output_path, output)

        print("\n=== MULTI-IF (ENGLISH) METRICS ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {output_path}")
        if failed_workers:
            raise SystemExit(
                f"{len(failed_workers)}/{num_workers} workers failed; "
                f"partial results saved to: {output_path}"
            )
        return output_path

    # ============================================================
    # Single GPU / CPU: keep serial evaluation
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model, device = modeling.load_beacon_model(
        model_path,
        device_id=int(cuda_id_list[0]) if cuda_id_list else cuda_id,
        num_sinks=num_sinks,
        strict_num_beacons=True,
    )

    all_eval_results = []
    errors = []

    def flush() -> None:
        metrics = multi_if_utils.compute_multi_if_metrics(all_eval_results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": int(cuda_id_list[0]) if cuda_id_list else cuda_id,
                "log_dir": resolved_log_dir,
                "num_splits": len(split_names),
                "splits": split_names,
                "total_english": total_english,
                "run_tag": run_tag,
                "num_beacons_per_segment": num_beacons_per_segment,
                "num_sinks": num_sinks,
                "attempted": len(all_eval_results) + len(errors),
                "succeeded": len(all_eval_results),
                "failed": len(errors),
            },
            "metrics": metrics,
            "results": all_eval_results,
            "errors": errors,
        }
        try:
            common.write_json_atomic(output_path, output)
        except Exception as e:
            print(f"[WARN] Failed to write {output_path}: {e}")

    flush()

    pbar = tqdm(total=total_english, desc="Evaluating Multi-IF (English)", unit="sample")
    try:
        global_english_idx = 0
        processed = 0
        for split in split_names:
            orig_indices = english_indices_by_split.get(split) or []
            for en_idx, orig_idx in enumerate(orig_indices):
                sample = ds[split][orig_idx]
                pbar.set_postfix(split=split, idx=en_idx)
                try:
                    generated_responses = run_multi_if_sample(
                        sample=sample,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        verbose=verbose,
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
                            "key": sample.get("key"),
                            "error": repr(e),
                        }
                    )
                finally:
                    global_english_idx += 1
                    processed += 1
                    pbar.update(1)
                    if flush_every > 0 and (processed % flush_every == 0):
                        flush()
    finally:
        pbar.close()

    flush()

    metrics = multi_if_utils.compute_multi_if_metrics(all_eval_results)
    print("\n=== MULTI-IF (ENGLISH) METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")

    return output_path


if __name__ == "__main__":
    try:
        import fire
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `fire`. Install with: pip install fire"
        ) from e

    fire.Fire(main)
