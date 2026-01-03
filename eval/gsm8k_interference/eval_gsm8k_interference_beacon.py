import json
import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from eval import common, modeling
import gsm8k_interference_utils as utils


# ============================================================
# 1. Worker Helpers
# ============================================================
def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"gsm8k_interference_beacon_{timestamp}_{run_tag}.worker{worker_id}.json"
    )


# ============================================================
# 2. Encode dialogue
# ============================================================
def encode_dialogue(
    dialogue: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    device: torch.device,
    enable_thinking: bool,
):
    text = utils.apply_chat_template(
        tokenizer,
        dialogue,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return modeling.encode_prompt(tokenizer, text, device)


# ============================================================
# 3. Generate response
# ============================================================
def generate_stream(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
):
    return modeling.generate_beacon_response(
        model,
        input_ids,
        attention_mask,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        apply_temperature_if_sampled=True,
    )


def _run_worker(
    *,
    worker_id: int,
    num_workers: int,
    cuda_id: int,
    model_path: str,
    num_sinks: int,
    gsm8k_path: Optional[str],
    gsm8k_split: str,
    ultrachat_path: str,
    seed: int,
    max_samples: Optional[int],
    history_max_turns: Optional[int],
    max_input_tokens: Optional[int],
    answer_instruction: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
    enable_thinking: bool,
    log_dir: str,
    timestamp: str,
    run_tag: str,
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if max_input_tokens is not None and tokenizer.model_max_length:
        if tokenizer.model_max_length < 100000 and max_input_tokens > tokenizer.model_max_length:
            max_input_tokens = int(tokenizer.model_max_length)

    model, device = modeling.load_beacon_model(
        model_path=model_path,
        device_id=cuda_id,
        num_sinks=num_sinks,
        tokenizer=tokenizer,
        strict_num_beacons=False,
    )
    num_beacons_per_segment = getattr(model.config, "num_beacons_per_segment", None)

    samples = utils.load_gsm8k(gsm8k_path, gsm8k_split)
    if max_samples is not None:
        samples = samples[: max_samples]

    total_samples = len(samples)
    offsets = utils.build_line_offsets(ultrachat_path)
    round_buckets = utils.build_round_buckets(ultrachat_path)
    plans = utils.build_history_plans(total_samples, history_max_turns, seed, round_buckets)
    worker_indices = list(range(worker_id, total_samples, num_workers))
    worker_plans = [plans[i] for i in worker_indices]
    histories, history_metas = utils.build_histories_from_plans(
        ultrachat_path, offsets, worker_plans
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
        metrics = utils.compute_gsm8k_metrics(results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "gsm8k_split": gsm8k_split,
                "run_tag": run_tag,
                "seed": seed,
                "max_samples": max_samples,
                "history_max_turns": history_max_turns,
                "max_input_tokens": max_input_tokens,
                "answer_instruction": answer_instruction,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
                "num_sinks": num_sinks,
                "num_beacons_per_segment": num_beacons_per_segment,
            },
            "metrics": metrics,
            "results": results,
            "errors": errors,
        }
        try:
            common.write_json_atomic(worker_out, output)
        except Exception as e:
            print(f"[WARN] Failed to write {worker_out}: {e}")

    flush()

    pbar = tqdm(total=len(worker_indices), desc=f"GSM8K (Beacon) worker{worker_id}", unit="sample", disable=not verbose)
    try:
        for local_idx, sample_idx in enumerate(worker_indices):
            sample = samples[sample_idx]
            history = histories[local_idx]
            history_meta = history_metas[local_idx]
            history_indices = history_meta.get("history_indices") or []
            pbar.set_postfix(idx=sample_idx)
            try:
                question = sample.get("question") or ""
                gold = utils.parse_gsm8k_answer(sample.get("answer") or "")

                messages = utils.prepare_messages(
                    history=history,
                    question=question,
                    answer_instruction=answer_instruction,
                    system_prompt=system_prompt,
                    history_max_turns=history_max_turns,
                )
                messages, _ = utils.truncate_to_max_tokens(
                    messages, tokenizer, max_input_tokens, enable_thinking
                )
                history_used = utils.extract_history_from_messages(messages)

                input_ids, attention_mask = encode_dialogue(
                    dialogue=messages,
                    tokenizer=tokenizer,
                    device=device,
                    enable_thinking=enable_thinking,
                )

                pred_text = generate_stream(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )
                pred = utils.extract_last_number(pred_text)
                is_correct = pred is not None and gold is not None and pred == gold

                history_turns = utils.count_history_turns(history_used)
                results.append(
                    {
                        "index": sample_idx,
                        "global_index": sample_idx,
                        "history_index": history_indices[0] if history_indices else None,
                        "history_indices": history_indices,
                        "history_rounds": history_meta.get("history_rounds") or [],
                        "history": history_used,
                        "history_len": len(history_used),
                        "history_messages": len(history_used),
                        "history_turns": history_turns,
                        "history_original_len": history_meta.get("history_source_messages"),
                        "history_original_turns": history_meta.get("history_source_turns"),
                        "question": question,
                        "gold": gold,
                        "prediction": pred,
                        "prediction_text": pred_text,
                        "correct": is_correct,
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "index": sample_idx,
                        "global_index": sample_idx,
                        "history_index": history_indices[0] if history_indices else None,
                        "history_indices": history_indices,
                        "error": repr(e),
                    }
                )
            finally:
                try:
                    if progress_counter is not None:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                except Exception:
                    pass
                if flush_every > 0 and (len(results) + len(errors)) % flush_every == 0:
                    flush()
                pbar.update(1)
    finally:
        pbar.close()

    flush()


def main(
    model_path: str,
    gsm8k_path: Optional[str] = None,
    gsm8k_split: str = "test",
    ultrachat_path: str = os.path.join(PROJECT_ROOT, "ultrachat-200k.jsonl"),
    seed: int = 42,
    max_samples: Optional[int] = None,
    history_max_turns: int = 6,
    max_input_tokens: int = 4096,
    answer_instruction: str = "",
    system_prompt: str = "You are a helpful assistant. Follow the user's instructions carefully and respond concisely.",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: Optional[bool] = None,
    enable_thinking: bool = False,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    run_tag: Optional[str] = None,
    verbose: bool = False,
    num_sinks: int = 0,
    flush_every: int = 50,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs", "gsm8k_interference")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if do_sample is None:
        do_sample = temperature > 0.0

    num_beacons_per_segment: Optional[int] = None
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_beacons_per_segment = getattr(cfg, "num_beacons_per_segment", None)
    except Exception as e:
        print(f"[WARN] Failed to read model config from {model_path}: {e}")

    if run_tag is None:
        beacon_part = (
            f"{int(num_beacons_per_segment)}beacon"
            if num_beacons_per_segment is not None
            else "unknownbeacon"
        )
        run_tag = f"{beacon_part}_{int(num_sinks)}sink"

    cuda_id_list = common.parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    common.validate_cuda_ids(cuda_id_list)

    samples = utils.load_gsm8k(gsm8k_path, gsm8k_split)
    if max_samples is not None:
        samples = samples[: max_samples]
    total_samples = len(samples)

    output_path = os.path.join(
        resolved_log_dir, f"gsm8k_interference_beacon_{timestamp}_{run_tag}.json"
    )

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
                    "gsm8k_path": gsm8k_path,
                    "gsm8k_split": gsm8k_split,
                    "ultrachat_path": ultrachat_path,
                    "seed": seed,
                    "max_samples": max_samples,
                    "history_max_turns": history_max_turns,
                    "max_input_tokens": max_input_tokens,
                    "answer_instruction": answer_instruction,
                    "system_prompt": system_prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "top_p": top_p,
                    "enable_thinking": enable_thinking,
                    "log_dir": resolved_log_dir,
                    "timestamp": timestamp,
                    "run_tag": run_tag,
                    "verbose": verbose,
                    "flush_every": flush_every,
                    "progress_counter": progress_counter,
                },
            )
            p.start()
            procs.append(p)

        pbar = tqdm(total=total_samples, desc="Evaluating GSM8K (Beacon)", unit="sample")
        last = 0
        try:
            while any(p.is_alive() for p in procs):
                try:
                    with progress_counter.get_lock():
                        current = int(progress_counter.value)
                except Exception:
                    current = last

                if current > last:
                    step = min(current - last, total_samples - last)
                    if step > 0:
                        pbar.update(step)
                        last += step

                time.sleep(0.2)

            try:
                with progress_counter.get_lock():
                    current = int(progress_counter.value)
            except Exception:
                current = last
            if current > last:
                step = min(current - last, total_samples - last)
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

        metrics = utils.compute_gsm8k_metrics(all_eval_results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_ids": cuda_id_list,
                "log_dir": resolved_log_dir,
                "num_workers": num_workers,
                "gsm8k_split": gsm8k_split,
                "total_samples": total_samples,
                "run_tag": run_tag,
                "attempted": len(all_eval_results) + len(errors),
                "succeeded": len(all_eval_results),
                "failed": len(errors),
                "worker_outputs": worker_paths,
                "failed_workers": failed_workers,
                "num_beacons_per_segment": num_beacons_per_segment,
                "num_sinks": num_sinks,
            },
            "metrics": metrics,
            "results": all_eval_results,
            "errors": errors,
        }
        common.write_json_atomic(output_path, output)

        print("\n=== GSM8K INTERFERENCE (BEACON) METRICS ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {output_path}")
        if failed_workers:
            raise SystemExit(
                f"{len(failed_workers)}/{num_workers} workers failed; "
                f"partial results saved to: {output_path}"
            )
        return output_path

    # ============================================================
    # Single GPU / CPU: serial evaluation
    # ============================================================
    _run_worker(
        worker_id=0,
        num_workers=1,
        cuda_id=int(cuda_id_list[0]) if cuda_id_list else cuda_id,
        model_path=model_path,
        num_sinks=num_sinks,
        gsm8k_path=gsm8k_path,
        gsm8k_split=gsm8k_split,
        ultrachat_path=ultrachat_path,
        seed=seed,
        max_samples=max_samples,
        history_max_turns=history_max_turns,
        max_input_tokens=max_input_tokens,
        answer_instruction=answer_instruction,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        enable_thinking=enable_thinking,
        log_dir=resolved_log_dir,
        timestamp=timestamp,
        run_tag=run_tag,
        verbose=verbose,
        flush_every=flush_every,
        progress_counter=None,
    )

    worker_out = _worker_output_path(resolved_log_dir, timestamp, run_tag, 0)
    if os.path.exists(worker_out):
        try:
            with open(worker_out, "r", encoding="utf-8") as f:
                output = json.load(f)
            common.write_json_atomic(output_path, output)
            print(f"\nSaved to: {output_path}")
        except Exception as e:
            print(f"[WARN] Failed to finalize output: {e}")
    return output_path


if __name__ == "__main__":
    try:
        import fire
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `fire`. Install with: pip install fire"
        ) from e

    fire.Fire(main)
