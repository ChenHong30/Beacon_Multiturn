import sys
import os
import re
import json
from datetime import datetime
from collections import defaultdict
from typing import Any, List, Dict, Optional

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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    log_dir: Optional[str] = None,
    verbose: bool = False,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(resolved_log_dir, f"multi_if_pipeline_{timestamp}.json")

    # 使用 pipeline 替代手动加载 model 和 tokenizer
    pipe, tokenizer, device = create_generation_pipeline(model_path, cuda_id)

    ds = load_dataset("facebook/Multi-IF")
    # 筛选英文样本
    ds_en = ds.filter(lambda x: str(x.get("language", "")).lower() == "english")

    split_names = list(ds_en.keys())
    total = sum(len(ds_en[s]) for s in split_names)

    all_eval_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
        """计算当前指标并保存到磁盘。"""
        metrics = compute_multi_if_metrics(all_eval_results)
        output = {
            "meta": {
                "timestamp": timestamp,
                "model_path": model_path,
                "cuda_id": cuda_id,
                "log_dir": resolved_log_dir,
                "num_splits": len(split_names),
                "splits": split_names,
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

    pbar = tqdm(total=total, desc="Evaluating Multi-IF (Pipeline)", unit="sample")
    try:
        for split in split_names:
            split_ds = ds_en[split]
            for idx in range(len(split_ds)):
                sample = split_ds[idx]
                pbar.set_postfix(split=split, idx=idx)
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
                    eval_result["index"] = idx
                    all_eval_results.append(eval_result)
                except Exception as e:
                    errors.append(
                        {
                            "split": split,
                            "index": idx,
                            "key": sample.get("key"),
                            "error": repr(e),
                        }
                    )
                finally:
                    pbar.update(1)
                    # 每次处理完一个样本都保存一次，以便实时查看进度和避免数据丢失
                    flush()
    finally:
        pbar.close()

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