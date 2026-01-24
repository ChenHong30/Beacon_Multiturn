# flake8: noqa: E501
"""
SafeDialBench evaluation script for Base model (without Beacon compression).

This script evaluates the safety of language models against multi-turn
jailbreaking attacks from the SafeDialBench dataset.

Evaluation dimensions:
1. Identification - Ability to identify safety risks
2. Handling - Ability to handle/respond to risks appropriately
3. Consistency - Consistency in maintaining safety across turns
"""
import json
import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared utilities
import eval.safediabench.eval_safediabench_beacon as beacon


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
    judge_client: Union[beacon.OpenAICompatClient, beacon.HKUSTGZJudgeClient],
    judge_prompts: Dict[str, Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
) -> Dict[str, Any]:
    """
    Evaluate a single SafeDialBench dialogue using base model.
    """
    dialogue_id = dialogue.get("id", dialogue_index)
    task = dialogue.get("task", "Ethics")
    method = dialogue.get("method", "")
    model_type = dialogue.get("model_type", "")
    scene = dialogue.get("scene", "")
    history = dialogue.get("history", [])

    # Get appropriate judge prompt for this task
    judge_type = beacon.TASK_TO_JUDGE_TYPE.get(task, "ethics-evaluation")
    judge_prompt = judge_prompts.get(judge_type)
    if not judge_prompt:
        judge_prompt = judge_prompts.get("ethics-evaluation")

    if not judge_prompt:
        raise ValueError(f"No judge prompt found for task: {task}")

    results = {
        "id": dialogue_id,
        "task": task,
        "method": method,
        "model_type": model_type,
        "scene": scene,
        "num_turns": len(history),
        "identification_score": 0.0,
        "handling_score": 0.0,
        "consistency_score": 0.0,
        "turns": [],
    }

    # Check if we should only evaluate the last turn
    evaluate_last_only = method in beacon.LAST_TURN_ONLY_METHODS

    # Generate responses for all turns
    # NOTE: Following the original SafeDialBench logic:
    # - Model generates response for each turn
    # - But the conversation history uses the GROUND TRUTH bot responses,
    #   not the model-generated responses (for fair evaluation)
    model_responses = []
    conversation: List[Dict[str, str]] = []
    system_prompt = dialogue.get("system")
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    for turn_index, turn in enumerate(history):
        user_content = turn.get("user", "")
        bot_content = turn.get("bot", "")  # Ground truth response
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
        model_responses.append(response)
        # Use GROUND TRUTH bot response for conversation history (as per original SafeDialBench)
        conversation.append({"role": "assistant", "content": bot_content})

    # Evaluate turns
    scores_list = []

    if evaluate_last_only:
        turn_index = len(history) - 1
        context = beacon.build_conversation_context(
            history, model_responses, len(history), use_model_response_for_last=True
        )
        scores, judgment = beacon.judge_dialogue_turn(judge_client, context, judge_prompt)
        scores_list.append(scores)

        results["turns"].append({
            "turn_index": turn_index,
            "user_content": history[turn_index].get("user", ""),
            "response": model_responses[turn_index],
            "scores": scores,
            "judgment": judgment,
        })
    else:
        for turn_index in range(len(history)):
            context = beacon.build_conversation_context(
                history, model_responses, turn_index + 1, use_model_response_for_last=True
            )
            scores, judgment = beacon.judge_dialogue_turn(judge_client, context, judge_prompt)
            scores_list.append(scores)

            results["turns"].append({
                "turn_index": turn_index,
                "user_content": history[turn_index].get("user", ""),
                "response": model_responses[turn_index],
                "scores": scores,
                "judgment": judgment,
            })

    # Calculate final scores
    if scores_list:
        identification_scores = [s[0] for s in scores_list]
        handling_scores = [s[1] for s in scores_list]
        consistency_scores = [s[2] for s in scores_list]

        results["identification_score"] = min(identification_scores)
        results["handling_score"] = min(handling_scores)
        results["consistency_score"] = min(consistency_scores)

    return results


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"safediabench_base_{timestamp}_{run_tag}.worker{worker_id}.json"
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
    judge_prompts: Dict[str, Dict[str, Any]],
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

    # Select judge client based on config provider
    judge_api_provider = api_config.get("judge_api_provider", "dashscope")
    if judge_api_provider == "hkustgz":
        judge_client = beacon.HKUSTGZJudgeClient(
            api_key=api_config["hkustgz_api_key"],
            model=api_config.get("hkustgz_model", "Qwen"),
            timeout=api_config.get("request_timeout", 120),
            max_retries=api_config.get("max_retries", 3),
            retry_sleep=api_config.get("retry_sleep", 1.0),
            temperature=api_config.get("temperature", 0.7),
            top_p=api_config.get("top_p", 1.0),
            max_tokens=api_config.get("max_tokens", 2048),
        )
    else:
        judge_client = beacon.OpenAICompatClient(
            base_url=api_config["openai_base_url"],
            api_key=api_config["openai_api_key"],
            model=api_config["judge_model"],
            timeout=api_config.get("request_timeout", 120),
            max_retries=api_config.get("max_retries", 3),
            retry_sleep=api_config.get("retry_sleep", 1.0),
            temperature=api_config.get("temperature", 0.7),
            top_p=api_config.get("top_p", 1.0),
            max_tokens=api_config.get("max_tokens", 2048),
        )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed_dialogues = 0

    def flush() -> None:
        if results:
            avg_identification = sum(r["identification_score"] for r in results) / len(results)
            avg_handling = sum(r["handling_score"] for r in results) / len(results)
            avg_consistency = sum(r["consistency_score"] for r in results) / len(results)
        else:
            avg_identification = avg_handling = avg_consistency = 0.0

        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "judge_api_provider": api_config.get("judge_api_provider", "dashscope"),
                "judge_model": api_config.get("hkustgz_model", "Qwen") if api_config.get("judge_api_provider") == "hkustgz" else api_config.get("judge_model"),
                "processed_dialogues": processed_dialogues,
                "failed_dialogues": len(errors),
            },
            "metrics": {
                "avg_identification": avg_identification,
                "avg_handling": avg_handling,
                "avg_consistency": avg_consistency,
                "total_count": len(results),
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
                    judge_prompts=judge_prompts,
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
                        "task": dialogue.get("task"),
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
    prompts_dir: Optional[str] = None,
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
    enable_sampling: bool = False,
    sample_size_a: int = 200,
    sample_size_b: int = 200,
    sample_seed: int = 42,
) -> str:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    api_config_path = config_path or os.path.join(
        PROJECT_ROOT, "eval", "safediabench", "safediabench_config.json"
    )
    api_config = beacon._load_api_config(api_config_path)

    # Load judge prompts
    resolved_prompts_dir = prompts_dir or os.path.join(
        PROJECT_ROOT, "eval", "safediabench", "judge_prompts"
    )
    judge_prompts = beacon.load_judge_prompts(resolved_prompts_dir)
    if not judge_prompts:
        raise ValueError(f"No judge prompts found in {resolved_prompts_dir}")

    run_tag = "base"
    output_path = os.path.join(
        resolved_log_dir, f"safediabench_base_{timestamp}_{run_tag}.json"
    )

    dialogues = beacon._load_safediabench_dataset(data_path)
    print(f"Loaded {len(dialogues)} dialogues from {data_path}")

    # Apply sampling if enabled
    if enable_sampling:
        dialogues = beacon._sample_dataset(
            dialogues,
            sample_size_a=sample_size_a,
            sample_size_b=sample_size_b,
            seed=sample_seed,
        )

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
                "judge_prompts": judge_prompts,
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
        pbar = tqdm(total=len(dialogues), desc="Evaluating SafeDialBench", unit="dialogue")

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
    if all_results:
        avg_identification = sum(r["identification_score"] for r in all_results) / len(all_results)
        avg_handling = sum(r["handling_score"] for r in all_results) / len(all_results)
        avg_consistency = sum(r["consistency_score"] for r in all_results) / len(all_results)
        avg_overall = (avg_identification + avg_handling + avg_consistency) / 3
    else:
        avg_identification = avg_handling = avg_consistency = avg_overall = 0.0

    # Calculate metrics by task
    task_metrics: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        task = r.get("task", "unknown")
        if task not in task_metrics:
            task_metrics[task] = {
                "identification_scores": [],
                "handling_scores": [],
                "consistency_scores": [],
                "count": 0,
            }
        task_metrics[task]["identification_scores"].append(r["identification_score"])
        task_metrics[task]["handling_scores"].append(r["handling_score"])
        task_metrics[task]["consistency_scores"].append(r["consistency_score"])
        task_metrics[task]["count"] += 1

    for task, metrics in task_metrics.items():
        count = metrics["count"]
        metrics["avg_identification"] = sum(metrics["identification_scores"]) / count
        metrics["avg_handling"] = sum(metrics["handling_scores"]) / count
        metrics["avg_consistency"] = sum(metrics["consistency_scores"]) / count
        metrics["avg_overall"] = (
            metrics["avg_identification"] + metrics["avg_handling"] + metrics["avg_consistency"]
        ) / 3
        del metrics["identification_scores"]
        del metrics["handling_scores"]
        del metrics["consistency_scores"]

    # Calculate metrics by method
    method_metrics: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        method = r.get("method", "unknown")
        if method not in method_metrics:
            method_metrics[method] = {
                "identification_scores": [],
                "handling_scores": [],
                "consistency_scores": [],
                "count": 0,
            }
        method_metrics[method]["identification_scores"].append(r["identification_score"])
        method_metrics[method]["handling_scores"].append(r["handling_score"])
        method_metrics[method]["consistency_scores"].append(r["consistency_score"])
        method_metrics[method]["count"] += 1

    for method, metrics in method_metrics.items():
        count = metrics["count"]
        metrics["avg_identification"] = sum(metrics["identification_scores"]) / count
        metrics["avg_handling"] = sum(metrics["handling_scores"]) / count
        metrics["avg_consistency"] = sum(metrics["consistency_scores"]) / count
        metrics["avg_overall"] = (
            metrics["avg_identification"] + metrics["avg_handling"] + metrics["avg_consistency"]
        ) / 3
        del metrics["identification_scores"]
        del metrics["handling_scores"]
        del metrics["consistency_scores"]

    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": final_num_workers,
            "num_dialogues": len(dialogues),
            "run_tag": run_tag,
            "judge_api_provider": api_config.get("judge_api_provider", "dashscope"),
            "judge_model": api_config.get("hkustgz_model", "Qwen") if api_config.get("judge_api_provider") == "hkustgz" else api_config.get("judge_model"),
            "attempted_items": len(all_results),
            "failed_dialogues": len(errors),
            "worker_outputs": worker_paths,
            "failed_workers": failed_workers,
        },
        "metrics": {
            "avg_identification": avg_identification,
            "avg_handling": avg_handling,
            "avg_consistency": avg_consistency,
            "avg_overall": avg_overall,
            "total_count": len(all_results),
            "by_task": task_metrics,
            "by_method": method_metrics,
        },
        "results": all_results,
        "errors": errors,
    }
    beacon.write_json_atomic(output_path, output)

    print("\n=== SafeDialBench METRICS ===")
    print(f"Overall Scores (avg):")
    print(f"  Identification: {avg_identification:.4f}")
    print(f"  Handling:       {avg_handling:.4f}")
    print(f"  Consistency:    {avg_consistency:.4f}")
    print(f"  Overall:        {avg_overall:.4f}")
    print(f"\nTotal dialogues evaluated: {len(all_results)}")
    print("\nBy Task:")
    for task, metrics in sorted(task_metrics.items()):
        print(f"  {task}: I={metrics['avg_identification']:.2f} H={metrics['avg_handling']:.2f} C={metrics['avg_consistency']:.2f} (n={metrics['count']})")
    print("\nBy Method:")
    for method, metrics in sorted(method_metrics.items()):
        print(f"  {method}: I={metrics['avg_identification']:.2f} H={metrics['avg_handling']:.2f} C={metrics['avg_consistency']:.2f} (n={metrics['count']})")
    print(f"\nSaved to: {output_path}")

    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{final_num_workers} workers failed; "
            f"partial results saved to: {output_path}"
        )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SafeDialBench Base evaluation.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument(
        "--prompts-dir",
        "--prompts_dir",
        dest="prompts_dir",
        default=None,
        help="Path to directory containing judge prompt JSONL files.",
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        default=None,
        help="Path to JSON config with API configuration.",
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
    # Sampling options
    parser.add_argument(
        "--enable-sampling", "--enable_sampling",
        dest="enable_sampling",
        action="store_true",
        default=False,
        help="Enable balanced sampling (200 samples each for case A and B by default).",
    )
    parser.add_argument("--sample-size-a", "--sample_size_a", dest="sample_size_a", type=int, default=200,
                        help="Number of samples for case A (last-turn-only methods).")
    parser.add_argument("--sample-size-b", "--sample_size_b", dest="sample_size_b", type=int, default=200,
                        help="Number of samples for case B (all-turns methods).")
    parser.add_argument("--sample-seed", "--sample_seed", dest="sample_seed", type=int, default=42,
                        help="Random seed for sampling.")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        prompts_dir=args.prompts_dir,
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
        enable_sampling=args.enable_sampling,
        sample_size_a=args.sample_size_a,
        sample_size_b=args.sample_size_b,
        sample_seed=args.sample_seed,
    )
