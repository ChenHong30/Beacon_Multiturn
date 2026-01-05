import sys
import os
import re
import json
import multiprocessing as mp
import time
from datetime import datetime
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple

# 导入 transformers pipeline
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, pipeline # 导入 pipeline

from tqdm.auto import tqdm

# Assuming metrics.py is available in your path as per original script
# 确保 'metrics.py' 中的 eval_multi_if_sample 函数可用
try:
    from metrics import eval_multi_if_sample
except ImportError:
    # 假设如果找不到，则使用一个mock函数，实际使用时需要确保该文件存在
    def eval_multi_if_sample(sample, generated_responses):
        print("[WARN] metrics.py not found. Using mock evaluation function.")
        # 返回一个包含基本信息的模拟结果
        return {
            "key": sample.get("key"),
            "turns": [
                {
                    "turn_id": i + 1, 
                    "follow_strict": [True], 
                    "follow_loose": [True],
                    "instruction_id_list": ["mock_id_0"],
                    "generated_response": resp
                } 
                for i, resp in enumerate(generated_responses)
            ]
        }


# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 1. Pipeline Setup (取代 load_standard_model)
# ============================================================
def create_generation_pipeline(model_path: str, device_id: int = 0):
    """
    创建 Hugging Face text-generation pipeline.
    """
    print(f"Creating generation pipeline for: {model_path}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # 确定设备 ID: 如果有 CUDA 则使用指定的 GPU ID，否则使用 CPU (-1)
    pipe_device = device_id if torch.cuda.is_available() else -1

    # 加载 Tokenizer (需要它来格式化 Chat Template)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 创建 text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        # 确保 dtype 正确，以节省 GPU 内存
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=pipe_device,
        model_kwargs={
            "attn_implementation": "eager",
        }
    )

    print(f"Pipeline created. Model loaded on {pipe.device} (tokenizer available)")
    # 返回 pipeline 对象、tokenizer（用于 chat template）和设备信息
    return pipe, tokenizer, pipe.device

def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """原子性地写入 JSON 文件，防止写入失败导致文件损坏。"""
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
        log_dir, f"multi_if_pipeline_{timestamp}_{run_tag}.worker{worker_id}.json"
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
    verbose: bool,
    flush_every: int,
    progress_counter: Optional[Any],
    task_queue: Any,
) -> None:
    worker_out = _worker_output_path(log_dir, timestamp, run_tag, worker_id)

    pipe, tokenizer, device = create_generation_pipeline(
        model_path=model_path,
        device_id=cuda_id,
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
                    pipe=pipe,
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
    finally:
        flush()


# ============================================================
# 2. Extract turns from Multi-IF sample (无变化)
# ============================================================
def extract_turns_from_multi_if(sample: Dict) -> List[Dict[str, str]]:
    """
    从原始 Multi-IF 样本中提取按顺序排列的对话回合。
    """
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
# 3. Run one Multi-IF sample (ALL turns) - 使用 Pipeline
# ============================================================
def run_multi_if_sample(
    sample: Dict,
    pipe,           # 接受 pipeline 对象
    tokenizer: AutoTokenizer,
    device,         # 只是为了保持签名一致，pipe 内部已管理设备
    *,
    verbose: bool = False,
):
    """
    运行一个 Multi-IF 样本的整个多轮对话，使用 pipeline 生成响应。
    """
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

        # 1. 使用 tokenizer 格式化整个对话历史为模型输入字符串
        prompt_text = tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        if verbose:
            print(f"Prompt text length: {len(prompt_text)} characters")

        # 2. 调用 pipeline 进行生成
        # pipeline 负责编码字符串、调用模型生成和解码结果
        try:
            output = pipe(
                prompt_text,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                
                # Pipeline 特有的参数
                return_full_text=False, # 确保只返回新生成的文本
                clean_up_tokenization_spaces=True,
            )
            
            # 提取生成的文本
            resp = output[0]['generated_text'].strip()
            
        except Exception as e:
            resp = f"[ERROR_GENERATION] {repr(e)}"
            print(f"[ERROR] Generation failed: {resp}")


        if verbose:
            print(f"[Assistant]\n{resp}")

        # 将模型的回复加入到对话历史中，作为下一轮的上下文
        dialogue.append({"role": "assistant", "content": resp})
        generated_responses.append(resp)

    return generated_responses


# ============================================================
# 4. Metrics aggregation (无变化)
# ============================================================
def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def compute_multi_if_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    聚合所有评估结果，计算不同粒度的指标。
    """
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
            # 确保这些键存在且是列表
            follow_strict = turn.get("follow_strict") or []
            follow_loose = turn.get("follow_loose") or []
            instruction_ids = turn.get("instruction_id_list") or []

            total_turns += 1

            # 回合通过：该回合中所有指令都通过
            strict_ok = bool(follow_strict) and all(bool(x) for x in follow_strict)
            loose_ok = bool(follow_loose) and all(bool(x) for x in follow_loose)

            strict_turn_pass += int(strict_ok)
            loose_turn_pass += int(loose_ok)

            # 对话通过：该对话中所有回合都必须通过
            conv_strict_ok = conv_strict_ok and strict_ok
            conv_loose_ok = conv_loose_ok and loose_ok

            # 统计指令级通过数
            total_instructions += len(follow_strict)
            strict_instruction_pass += sum(int(bool(x)) for x in follow_strict)
            loose_instruction_pass += sum(int(bool(x)) for x in follow_loose)

            # 统计按指令 ID 分类的指标
            for instruction_id, fs, fl in zip(instruction_ids, follow_strict, follow_loose):
                rec = by_instruction[str(instruction_id)]
                rec["total"] += 1
                rec["strict_pass"] += int(bool(fs))
                rec["loose_pass"] += int(bool(fl))

        strict_conv_pass += int(conv_strict_ok)
        loose_conv_pass += int(conv_loose_ok)

    # 计算指令级准确率
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
# 5. Main: evaluate all English samples
# ============================================================
def main(
    model_path: str,
    cuda_id: int = 0,
    cuda_ids: Optional[str] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    flush_every: int = 1,
    num_workers: int = 16,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "base"
    output_path = os.path.join(resolved_log_dir, f"multi_if_pipeline_{timestamp}_{run_tag}.json")

    cuda_id_list = _parse_cuda_ids(cuda_id=cuda_id, cuda_ids=cuda_ids)
    _validate_cuda_ids(cuda_id_list)

    ds = load_dataset("facebook/Multi-IF")
    split_names, english_indices_by_split = _get_english_indices_by_split(ds)
    total_english = sum(len(v) for v in english_indices_by_split.values())

    # ============================================================
    # Multi-GPU: one process per GPU, shard by global_index % num_workers
    # ============================================================
    if torch.cuda.is_available() and len(cuda_id_list) > 1:
        # Prepare Queue
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()

        print(f"Generating tasks for {total_english} samples...")
        global_english_idx = 0
        for split in split_names:
            orig_indices = english_indices_by_split.get(split) or []
            for en_idx, orig_idx in enumerate(orig_indices):
                # Task: (split, en_idx, orig_idx, global_english_idx)
                task_queue.put((split, en_idx, orig_idx, global_english_idx))
                global_english_idx += 1

        final_num_workers = num_workers
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
            cid = cuda_id_list[wid % len(cuda_id_list)]
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
                    "verbose": verbose,
                    "flush_every": flush_every,
                    "progress_counter": progress_counter,
                    "task_queue": task_queue,
                },
            )
            p.start()
            procs.append(p)

        pbar = tqdm(total=total_english, desc="Evaluating Multi-IF (Pipeline)", unit="sample")
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
                        "cuda_id": cuda_id_list[wid % len(cuda_id_list)],
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

        metrics = compute_multi_if_metrics(all_eval_results)
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

        print("\n=== MULTI-IF (PIPELINE) METRICS ===")
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
    pipe, tokenizer, device = create_generation_pipeline(
        model_path,
        int(cuda_id_list[0]) if cuda_id_list else cuda_id,
    )

    all_eval_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
        """计算当前指标并保存到磁盘。"""
        metrics = compute_multi_if_metrics(all_eval_results)
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
                "attempted": len(all_eval_results) + len(errors),
                "succeeded": len(all_eval_results),
                "failed": len(errors),
            },
            "metrics": metrics,
            "results": all_eval_results,
            "errors": errors,
        }
        try:
            write_json_atomic(output_path, output)
        except Exception as e:
            print(f"[WARN] Failed to write {output_path}: {e}")

    # 首次保存，确保文件存在
    flush()

    pbar = tqdm(total=total_english, desc="Evaluating Multi-IF (Pipeline)", unit="sample")
    try:
        global_english_idx = 0
        processed = 0
        for split in split_names:
            orig_indices = english_indices_by_split.get(split) or []
            for en_idx, orig_idx in enumerate(orig_indices):
                sample = ds[split][orig_idx]
                pbar.set_postfix(split=split, idx=en_idx)
                try:
                    # 将 pipe 传入 run_multi_if_sample
                    generated_responses = run_multi_if_sample(
                        sample=sample,
                        pipe=pipe,
                        tokenizer=tokenizer,
                        device=device,
                        verbose=verbose,
                    )

                    # 评估生成的响应
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
    metrics = compute_multi_if_metrics(all_eval_results)
    print("\n=== MULTI-IF (PIPELINE) METRICS ===")
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

    # 使用 fire 允许通过命令行参数运行 main 函数
    # 示例运行命令: python your_script.py --model_path "your/model/path" --cuda_id 0
    fire.Fire(main)
