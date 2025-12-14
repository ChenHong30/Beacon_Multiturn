import sys
import os
import re
import json
from datetime import datetime
from collections import defaultdict
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
from modeling_qwen3 import Qwen3ForCausalLM as BeaconQwen3ForCausalLM


# ============================================================
# 1. Load Beacon model
# ============================================================
def load_beacon_model(model_path: str, device_id: int = 0, num_sinks: int = 1):
    print(f"Loading Beacon model: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    config.num_sinks = num_sinks

    print(f"num_sinks = {config.num_sinks}")
    print(f"num_beacons_per_segment = {config.num_beacons_per_segment}")

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model = BeaconQwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    max_new_tokens: int = 2048
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
            tokenizer
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
    log_dir: Optional[str] = None,
    verbose: bool = False,
    num_sinks: int = 1,
):
    resolved_log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(resolved_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(resolved_log_dir, f"multi_if_beacon_{timestamp}.json")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    model, device = load_beacon_model(
        model_path,
        device_id=cuda_id,
        num_sinks=num_sinks
    )

    ds = load_dataset("facebook/Multi-IF")
    ds_en = ds.filter(lambda x: str(x.get("language", "")).lower() == "english")

    split_names = list(ds_en.keys())
    total = sum(len(ds_en[s]) for s in split_names)

    all_eval_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def flush() -> None:
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

    flush()

    pbar = tqdm(total=total, desc="Evaluating Multi-IF (English)", unit="sample")
    try:
        for split in split_names:
            split_ds = ds_en[split]
            for idx in range(len(split_ds)):
                sample = split_ds[idx]
                pbar.set_postfix(split=split, idx=idx)
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
                    flush()
    finally:
        pbar.close()

    metrics = compute_multi_if_metrics(all_eval_results)
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
