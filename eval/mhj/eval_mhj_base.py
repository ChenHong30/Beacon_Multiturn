# flake8: noqa: E501
"""
MHJ (Multi-turn Human Jailbreaking) evaluation script for Base model.

This script evaluates the safety/robustness of language models against multi-turn
jailbreaking attacks from the MHJ dataset.

Evaluation logic:
1. For each sample, generate responses turn by turn
2. After each turn, use a judge model to evaluate if the response is harmful/jailbroken
3. If any turn is judged as jailbroken, mark the entire attack as successful and stop
4. Calculate Attack Success Rate (ASR) across all samples
"""
import json
import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared utilities
import eval.mhj.eval_mhj_beacon as beacon


def create_generation_pipeline(model_path: str, device_id: int = 0):
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
    return pipe, tokenizer


def generate_response(
    pipe,
    tokenizer,
    dialogue_messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        dialogue_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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


def _evaluate_dialogue(
    dialogue: Dict[str, Any],
    dialogue_index: int,
    pipe,
    tokenizer,
    judge_client: beacon.OpenAICompatClient,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
) -> Dict[str, Any]:
    """
    Evaluate a single MHJ dialogue (multi-turn jailbreak attempt).

    For each turn:
    1. Generate model response
    2. Judge if response is jailbroken
    3. If jailbroken, stop and mark attack as successful
    """
    dialogue_id = dialogue.get("id", dialogue_index)
    source = dialogue.get("source", "")
    tactic = dialogue.get("tactic", "")
    submission_message = dialogue.get("submission_message", "")
    system_prompt = dialogue.get("system")
    user_turns = dialogue.get("history", [])

    results = {
        "id": dialogue_id,
        "source": source,
        "tactic": tactic,
        "submission_message": submission_message,
        "num_turns": len(user_turns),
        "jailbroken": False,
        "jailbreak_turn": None,
        "turns": [],
    }

    # Build conversation incrementally
    conversation: List[Dict[str, str]] = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    for turn_index, turn in enumerate(user_turns):
        user_content = turn.get("content", "")
        conversation.append({"role": "user", "content": user_content})

        # Generate model response
        response = generate_response(
            pipe,
            tokenizer,
            conversation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )

        # Add response to conversation for next turn
        conversation.append({"role": "assistant", "content": response})

        # Judge the response
        judge_result = beacon.judge_response(
            judge_client,
            user_request=user_content,
            response=response,
        )

        turn_result = {
            "turn_index": turn_index,
            "user_content": user_content,
            "response": response,
            "judge_result": judge_result,
        }
        results["turns"].append(turn_result)

        # If jailbroken, stop evaluation for this dialogue
        if judge_result["jailbroken"]:
            results["jailbroken"] = True
            results["jailbreak_turn"] = turn_index
            break

    return results


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"mhj_base_{timestamp}_{run_tag}.worker{worker_id}.json"
    )


def _run_worker(
    *,
    worker_id: int,
    num_workers: int,
    cuda_id: int,
    model_path: str,
    log_dir: str,
    timestamp: str,
    run_tag: str,
    api_config: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
    flush_every: int,
    progress_counter: Optional[Any],
    task_queue: Any,
) -> None:
    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    pipe, tokenizer = create_generation_pipeline(
        model_path=model_path,
        device_id=cuda_id,
    )
    judge_client = beacon.OpenAICompatClient(
        base_url=api_config["openai_base_url"],
        api_key=api_config["openai_api_key"],
        model=api_config["judge_model"],
        timeout=api_config.get("request_timeout", 120),
        max_retries=api_config.get("max_retries", 3),
        retry_sleep=api_config.get("retry_sleep", 1.0),
        temperature=api_config.get("temperature", 0.0),
        top_p=api_config.get("top_p", 1.0),
        max_tokens=api_config.get("max_tokens", 512),
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed_dialogues = 0

    def flush() -> None:
        jailbroken_count = sum(1 for r in results if r.get("jailbroken", False))
        total_count = len(results)
        asr = jailbroken_count / total_count if total_count > 0 else 0.0

        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "judge_model": api_config["judge_model"],
                "processed_dialogues": processed_dialogues,
                "failed_dialogues": len(errors),
            },
            "metrics": {
                "asr": asr,
                "jailbroken_count": jailbroken_count,
                "total_count": total_count,
            },
            "results": results,
            "errors": errors,
        }
        try:
            beacon.write_json_atomic(worker_out, output)
        except Exception as e:
            print(f"[WARN] Failed to write {worker_out}: {e}")

    flush()

    processed = 0
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            idx, dialogue = task

            try:
                dialog_result = _evaluate_dialogue(
                    dialogue=dialogue,
                    dialogue_index=idx,
                    pipe=pipe,
                    tokenizer=tokenizer,
                    judge_client=judge_client,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )
                results.append(dialog_result)
            except Exception as e:
                errors.append(
                    {
                        "dialogue_index": idx,
                        "id": dialogue.get("id"),
                        "error": repr(e),
                    }
                )
            finally:
                processed += 1
                processed_dialogues += 1
                if progress_counter is not None:
                    try:
                        with progress_counter.get_lock():
                            progress_counter.value += 1
                    except Exception:
                        pass
                if flush_every > 0 and (processed % flush_every == 0):
                    flush()
    finally:
        flush()


def main(
    model_path: str,
    data_path: str,
    config_path: Optional[str] = None,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: Optional[float] = None,
    flush_every: int = 1,
    num_workers: int = 1,
) -> str:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    api_config_path = config_path or os.path.join(
        PROJECT_ROOT, "eval", "mhj", "mhj_config.json"
    )
    api_config = beacon._load_api_config(api_config_path)

    run_tag = "base"
    output_path = os.path.join(
        resolved_log_dir, f"mhj_base_{timestamp}_{run_tag}.json"
    )

    dialogues = beacon._load_mhj_dataset(data_path)
    print(f"Loaded {len(dialogues)} dialogues from {data_path}")

    cuda_id_list = beacon._parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    beacon._validate_cuda_ids(cuda_id_list)

    final_num_workers = num_workers if (torch.cuda.is_available() and len(cuda_id_list) > 0) else 1
    if not torch.cuda.is_available():
        final_num_workers = 1

    print(f"Plan: {final_num_workers} workers on devices: {cuda_id_list or 'CPU'}")

    # Prepare Queue
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()

    for idx, dialogue in enumerate(dialogues):
        task_queue.put((idx, dialogue))

    for _ in range(final_num_workers):
        task_queue.put(None)

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
                "log_dir": resolved_log_dir,
                "timestamp": timestamp,
                "run_tag": run_tag,
                "api_config": api_config,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "flush_every": flush_every,
                "progress_counter": progress_counter,
                "task_queue": task_queue,
            },
        )
        p.start()
        procs.append(p)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(dialogues), desc="Evaluating MHJ", unit="dialogue")

    last = 0
    try:
        while any(p.is_alive() for p in procs):
            try:
                with progress_counter.get_lock():
                    current = int(progress_counter.value)
            except Exception:
                current = last
            if current > last and pbar is not None:
                step = min(current - last, len(dialogues) - last)
                if step > 0:
                    pbar.update(step)
                    last += step
            time.sleep(0.5)

        try:
            with progress_counter.get_lock():
                current = int(progress_counter.value)
        except Exception:
            current = last
        if current > last and pbar is not None:
            step = min(current - last, len(dialogues) - last)
            if step > 0:
                pbar.update(step)
                last += step
    finally:
        if pbar is not None:
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

    # Merge results from all workers
    all_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for wp in worker_paths:
        if not os.path.exists(wp):
            print(f"[WARN] Missing worker output: {wp}")
            continue
        try:
            with open(wp, "r", encoding="utf-8") as f:
                d = json.load(f)
            all_results.extend(d.get("results") or [])
            errors.extend(d.get("errors") or [])
        except Exception as e:
            print(f"[WARN] Failed to read {wp}: {e}")

    # Calculate final metrics
    jailbroken_count = sum(1 for r in all_results if r.get("jailbroken", False))
    total_count = len(all_results)
    asr = jailbroken_count / total_count if total_count > 0 else 0.0

    # Calculate metrics by tactic
    tactic_metrics: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        tactic = r.get("tactic", "unknown")
        if tactic not in tactic_metrics:
            tactic_metrics[tactic] = {"jailbroken": 0, "total": 0}
        tactic_metrics[tactic]["total"] += 1
        if r.get("jailbroken", False):
            tactic_metrics[tactic]["jailbroken"] += 1

    for tactic, metrics in tactic_metrics.items():
        metrics["asr"] = metrics["jailbroken"] / metrics["total"] if metrics["total"] > 0 else 0.0

    # Calculate metrics by source
    source_metrics: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        source = r.get("source", "unknown")
        if source not in source_metrics:
            source_metrics[source] = {"jailbroken": 0, "total": 0}
        source_metrics[source]["total"] += 1
        if r.get("jailbroken", False):
            source_metrics[source]["jailbroken"] += 1

    for source, metrics in source_metrics.items():
        metrics["asr"] = metrics["jailbroken"] / metrics["total"] if metrics["total"] > 0 else 0.0

    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": final_num_workers,
            "num_dialogues": len(dialogues),
            "run_tag": run_tag,
            "judge_model": api_config["judge_model"],
            "attempted_items": len(all_results),
            "failed_dialogues": len(errors),
            "worker_outputs": worker_paths,
            "failed_workers": failed_workers,
        },
        "metrics": {
            "asr": asr,
            "jailbroken_count": jailbroken_count,
            "total_count": total_count,
            "by_tactic": tactic_metrics,
            "by_source": source_metrics,
        },
        "results": all_results,
        "errors": errors,
    }
    beacon.write_json_atomic(output_path, output)

    print("\n=== MHJ METRICS ===")
    print(f"Attack Success Rate (ASR): {asr:.4f} ({jailbroken_count}/{total_count})")
    print("\nBy Tactic:")
    for tactic, metrics in sorted(tactic_metrics.items()):
        print(f"  {tactic}: {metrics['asr']:.4f} ({metrics['jailbroken']}/{metrics['total']})")
    print("\nBy Source:")
    for source, metrics in sorted(source_metrics.items()):
        print(f"  {source}: {metrics['asr']:.4f} ({metrics['jailbroken']}/{metrics['total']})")
    print(f"\nSaved to: {output_path}")

    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{final_num_workers} workers failed; "
            f"partial results saved to: {output_path}"
        )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MHJ Base evaluation.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        default=None,
        help="Path to JSON config with openai_api_key/openai_base_url/judge_model.",
    )
    parser.add_argument("--cuda-id", "--cuda_id", dest="cuda_id", type=int, default=0)
    parser.add_argument("--cuda-ids", "--cuda_ids", dest="cuda_ids", default=None)
    parser.add_argument("--log-dir", "--log_dir", dest="log_dir", default=None)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", type=lambda x: str(x).lower() != "false", default=False)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=None)
    parser.add_argument("--flush-every", "--flush_every", dest="flush_every", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        config_path=args.config_path,
        cuda_id=args.cuda_id,
        cuda_ids=args.cuda_ids,
        log_dir=args.log_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        flush_every=args.flush_every,
        num_workers=args.num_workers,
    )
