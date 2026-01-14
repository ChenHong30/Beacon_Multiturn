import json
import os
import re
import sys
import time
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, pipeline
from tqdm.auto import tqdm

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import coreference_resolution_utils as utils


# ============================================================
# 1. Pipeline Setup (Base model)
# ============================================================
def create_generation_pipeline(model_path: str, device_id: int = 0):
    print(f"Creating generation pipeline for: {model_path}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    pipe_device = device_id if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=pipe_device,
        model_kwargs={
            "attn_implementation": "eager",
        },
    )
    print(f"Pipeline created. Model loaded on {pipe.device} (tokenizer available)")
    return pipe, tokenizer, pipe.device


def generate_response(
    pipe,
    tokenizer,
    dialogue_messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
    enable_thinking: bool,
) -> str:
    prompt_text = utils.apply_chat_template(
        tokenizer,
        dialogue_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_full_text": False,
        "clean_up_tokenization_spaces": True,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    output = pipe(prompt_text, **gen_kwargs)
    return output[0]["generated_text"].strip()


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _parse_cuda_ids(cuda_id: Any, cuda_ids: Optional[Any]) -> List[int]:
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


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"coreference_resolution_base_{timestamp}_{run_tag}.worker{worker_id}.json"
    )


def _run_worker(
    *,
    worker_id: int,
    num_workers: int,
    cuda_id: int,
    model_path: str,
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
    task_queue: Any,
    max_input_tokens: int,
) -> None:
    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    pipe, tokenizer, _device = create_generation_pipeline(
        model_path=model_path,
        device_id=cuda_id,
    )

    if max_input_tokens is not None and tokenizer.model_max_length:
        if tokenizer.model_max_length < 100000 and max_input_tokens > tokenizer.model_max_length:
            max_input_tokens = int(tokenizer.model_max_length)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
        metrics = utils.compute_coreference_metrics(results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "max_input_tokens": max_input_tokens,
                "answer_instruction": answer_instruction,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
                "attempted": len(results) + len(errors),
                "succeeded": len(results),
                "failed": len(errors),
            },
            "metrics": metrics,
            "results": results,
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

            sample_idx, sample = task

            try:
                raw_conversation = sample.get("messages") or []
                conversation = utils.clean_conversation(raw_conversation)

                messages: List[Dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(conversation)
                messages = utils.append_answer_instruction(messages, answer_instruction)

                messages, input_ids = utils.truncate_to_max_tokens(
                    messages, tokenizer, max_input_tokens, enable_thinking
                )
                conversation_used = [
                    {"role": msg.get("role"), "content": msg.get("content")}
                    for msg in messages
                    if msg.get("role") != "system"
                ]

                pred_text = generate_response(
                    pipe=pipe,
                    tokenizer=tokenizer,
                    dialogue_messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )

                ground_truth = sample.get("ground_truth")
                is_correct = utils.is_fuzzy_match(pred_text, ground_truth)

                final_question = utils.extract_last_user_message(conversation_used)
                results.append(
                    {
                        "index": sample_idx,
                        "dataset": sample.get("dataset"),
                        "id": sample.get("id"),
                        "meta_scenario": sample.get("meta_scenario"),
                        "target_key": sample.get("target_key"),
                        "num_distraction_rounds": sample.get("num_distraction_rounds"),
                        "final_question": final_question,
                        "conversation": conversation_used,
                        "conversation_len": len(conversation_used),
                        "conversation_original_len": len(conversation),
                        "conversation_truncated": len(conversation_used) < len(conversation),
                        "prompt_tokens": len(input_ids),
                        "ground_truth": ground_truth,
                        "prediction_text": pred_text,
                        "correct": is_correct,
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "index": sample_idx,
                        "dataset": sample.get("dataset"),
                        "id": sample.get("id"),
                        "error": repr(e),
                    }
                )
            finally:
                processed += 1
                try:
                    if progress_counter is not None:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                except Exception:
                    pass
                if flush_every > 0 and processed % flush_every == 0:
                    flush()
    finally:
        flush()


def main(
    model_path: str,
    coref_path: Optional[str] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
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
    run_tag: str = "base",
    verbose: bool = False,
    flush_every: int = 50,
    num_workers: int = 16,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs", "coreference_resolution")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if do_sample is None:
        do_sample = temperature > 0.0

    cuda_id_list = _parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    _validate_cuda_ids(cuda_id_list)

    samples = utils.load_coreference_dataset(coref_path)
    if max_samples is not None:
        samples = samples[: max_samples]
    total_samples = len(samples)

    output_path = os.path.join(
        resolved_log_dir, f"coreference_resolution_base_{timestamp}_{run_tag}.json"
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    for i in range(total_samples):
        task_queue.put((i, samples[i]))

    final_num_workers = num_workers if (torch.cuda.is_available() and len(cuda_id_list) > 0) else 1
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
                "task_queue": task_queue,
                "max_input_tokens": max_input_tokens,
            },
        )
        p.start()
        procs.append(p)

    pbar = tqdm(total=total_samples, desc="Evaluating Coreference Resolution (Base)", unit="sample")
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

            time.sleep(0.5)

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
                    "cuda_id": cuda_id_list[wid % len(cuda_id_list)] if cuda_id_list else 0,
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

    all_eval_results.sort(key=lambda r: r.get("index", 0))
    errors.sort(key=lambda r: r.get("index", 0))

    metrics = utils.compute_coreference_metrics(all_eval_results)
    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": num_workers,
            "coref_path": coref_path,
            "total_samples": total_samples,
            "run_tag": run_tag,
            "answer_instruction": answer_instruction,
            "system_prompt": system_prompt,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "enable_thinking": enable_thinking,
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

    print("\n=== COREFERENCE RESOLUTION (BASE) METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")
    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{num_workers} workers failed; "
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
