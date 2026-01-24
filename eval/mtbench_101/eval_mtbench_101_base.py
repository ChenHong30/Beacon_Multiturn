# flake8: noqa: E501
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

import eval_mtbench_101_beacon as beacon


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
    dialogue_index: Optional[int],
    pipe,
    tokenizer,
    judge_client: beacon.OpenAICompatClient,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
) -> List[Dict[str, Any]]:
    task = dialogue.get("task", "")
    multi_id = dialogue.get("id", None)
    turns = dialogue.get("history") or []
    skip_first = task in beacon.skip_first_tasks

    results: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(turns):
        if skip_first and turn_index == 0:
            continue

        dialogue_messages: List[Dict[str, str]] = []
        system_prompt = dialogue.get("system") or dialogue.get("system_prompt")
        if system_prompt:
            dialogue_messages.append({"role": "system", "content": str(system_prompt)})
        for i in range(turn_index + 1):
            user = beacon._get_turn_user(turns[i])
            dialogue_messages.append({"role": "user", "content": user})
            if i < turn_index:
                bot = beacon._get_turn_bot(turns[i])
                if bot:
                    dialogue_messages.append({"role": "assistant", "content": bot})

        pred = generate_response(
            pipe,
            tokenizer,
            dialogue_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )

        history_str = beacon._build_history_str(turns, turn_index)
        ref_answer = beacon._extract_reference(dialogue, turn, turn_index) if task in beacon.need_ref_tasks else ""
        system_prompt, prompt_template = beacon.eval_prompt_construct(task, ref_answer, history_str)
        prompt = prompt_template.format(prediction=pred)

        judge_text = judge_client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        judged = beacon.post_process_mtbench101({"prediction": judge_text})
        score = judged["score"] if judged else None

        results.append(
            {
                "task": task,
                "multi_id": multi_id,
                "turn_id": str(turn_index + 1),
                "dialogue_index": dialogue_index,
                "prediction": pred,
                "judge_prediction": judge_text,
                "score": score,
                "judge": {
                    "task": task,
                    "multi_id": multi_id,
                    "turn_id": str(turn_index + 1),
                },
            }
        )
    return results


def _worker_output_path(log_dir: str, timestamp: str, run_tag: str, worker_id: int) -> str:
    return os.path.join(
        log_dir, f"mtbench_101_base_{timestamp}_{run_tag}.worker{worker_id}.json"
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
    # Select judge client based on config provider
    judge_api_provider = api_config.get("judge_api_provider", "dashscope")
    if judge_api_provider == "hkustgz":
        judge_client = beacon.HKUSTGZJudgeClient(
            api_key=api_config["hkustgz_api_key"],
            model=api_config.get("hkustgz_model", "Qwen"),
            timeout=api_config.get("request_timeout", 120),
            max_retries=api_config.get("max_retries", 3),
            retry_sleep=api_config.get("retry_sleep", 1.0),
            temperature=api_config.get("temperature", 0.0),
            top_p=api_config.get("top_p", 1.0),
            max_tokens=api_config.get("max_tokens", 512),
        )
    else:
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
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "run_tag": run_tag,
                "judge_api_provider": api_config.get("judge_api_provider", "dashscope"),
                "judge_model": api_config.get("hkustgz_model", "Qwen") if api_config.get("judge_api_provider") == "hkustgz" else api_config["judge_model"],
                "processed_dialogues": processed_dialogues,
                "attempted_items": len(results),
                "failed_dialogues": len(errors),
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
                dialog_results = _evaluate_dialogue(
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
                results.extend(dialog_results)
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
    name: Optional[str] = None,
    config_path: Optional[str] = None,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: Optional[float] = None,
    flush_every: int = 1,
    num_workers: int = 16,
) -> str:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    api_config_path = config_path or os.path.join(
        PROJECT_ROOT, "eval", "mtbench_101", "mtbench_101_config.json"
    )
    api_config = beacon._load_api_config(api_config_path)

    run_tag = "base"
    output_path = os.path.join(
        resolved_log_dir, f"mtbench_101_base_{timestamp}_{run_tag}.json"
    )

    conversations = beacon._load_mtbench101(data_path, name=name)

    cuda_id_list = beacon._parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    beacon._validate_cuda_ids(cuda_id_list)

    final_num_workers = num_workers if (torch.cuda.is_available() and len(cuda_id_list) > 0) else 1
    if not torch.cuda.is_available():
        final_num_workers = 1

    print(f"Plan: {final_num_workers} workers on devices: {cuda_id_list or 'CPU'}")

    # Prepare Queue
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    
    for idx, dialogue in enumerate(conversations):
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
        pbar = tqdm(total=len(conversations), desc="Evaluating MTBench_101", unit="dialogue")
    
    last = 0
    try:
        while any(p.is_alive() for p in procs):
            try:
                with progress_counter.get_lock():
                    current = int(progress_counter.value)
            except Exception:
                current = last
            if current > last and pbar is not None:
                step = min(current - last, len(conversations) - last)
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
            step = min(current - last, len(conversations) - last)
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

    all_results.sort(key=lambda r: (r.get("dialogue_index"), r.get("turn_id")))

    judged_answers = [{"score": r.get("score")} for r in all_results if r.get("score") is not None]
    references = [r.get("judge") for r in all_results if r.get("score") is not None]
    results = beacon.get_final_results(judged_answers, references) if judged_answers else {}

    output = {
        "meta": {
            "timestamp": timestamp,
            "model_path": model_path,
            "cuda_ids": cuda_id_list,
            "log_dir": resolved_log_dir,
            "num_workers": final_num_workers,
            "num_dialogues": len(conversations),
            "run_tag": run_tag,
            "judge_api_provider": api_config.get("judge_api_provider", "dashscope"),
            "judge_model": api_config.get("hkustgz_model", "Qwen") if api_config.get("judge_api_provider") == "hkustgz" else api_config["judge_model"],
            "attempted_items": len(all_results),
            "failed_dialogues": len(errors),
            "worker_outputs": worker_paths,
            "failed_workers": failed_workers,
        },
        "metrics": results,
        "results": all_results,
        "errors": errors,
    }
    beacon.write_json_atomic(output_path, output)

    print("\n=== MTBench_101 METRICS ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")
    if failed_workers:
        raise SystemExit(
            f"{len(failed_workers)}/{final_num_workers} workers failed; "
            f"partial results saved to: {output_path}"
        )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MTBench_101 base evaluation.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--name", default=None, help="Dataset name when data-path is a directory.")
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
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=None)
    parser.add_argument("--flush-every", "--flush_every", dest="flush_every", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=16)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        name=args.name,
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
