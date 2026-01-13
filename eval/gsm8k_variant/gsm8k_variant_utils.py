import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number_str(value: str) -> str:
    value = value.strip()
    if not value:
        return value

    sign = ""
    if value[0] in "+-":
        sign = value[0]
        value = value[1:]

    if "." in value:
        integer, frac = value.split(".", 1)
        frac = frac.rstrip("0")
        if frac:
            value = f"{integer}.{frac}"
        else:
            value = integer

    integer = value
    frac = None
    if "." in value:
        integer, frac = value.split(".", 1)

    integer = integer.lstrip("0") or "0"
    value = f"{integer}.{frac}" if frac else integer

    if value == "0":
        sign = ""
    return f"{sign}{value}"


def apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_gsm8k_variant(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        path = os.path.join(os.path.dirname(__file__), "gsm8k_variant_dataset.jsonl")

    if path.endswith(".jsonl"):
        data = read_jsonl(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported GSM8K variant data format: {path}")


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    value = matches[-1].strip()
    return normalize_number_str(value)


def parse_gsm8k_answer(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    return extract_last_number(answer_text)


def normalize_role(role: Optional[str]) -> str:
    role = (role or "user").lower()
    if role not in {"user", "assistant", "system"}:
        return "user"
    return role


def clean_conversation(conversation: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in conversation or []:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role"))
        content = str(msg.get("content") or "")
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def extract_last_user_message(conversation: List[Dict[str, str]]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def append_answer_instruction(
    messages: List[Dict[str, str]],
    answer_instruction: str,
) -> List[Dict[str, str]]:
    if not answer_instruction:
        return messages

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            content = messages[idx].get("content") or ""
            messages[idx]["content"] = f"{content.strip()}\n\n{answer_instruction}"
            return messages

    messages.append({"role": "user", "content": answer_instruction})
    return messages


def truncate_to_max_tokens(
    messages: List[Dict[str, str]],
    tokenizer,
    max_input_tokens: Optional[int],
    enable_thinking: bool,
) -> Tuple[List[Dict[str, str]], List[int]]:
    if not max_input_tokens:
        prompt = apply_chat_template(tokenizer, messages, True, enable_thinking)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        return messages, ids

    trimmed = list(messages)
    while True:
        prompt = apply_chat_template(tokenizer, trimmed, True, enable_thinking)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(ids) <= max_input_tokens:
            return trimmed, ids
        remove_idx = 1 if trimmed and trimmed[0]["role"] == "system" else 0
        if len(trimmed) <= remove_idx + 1:
            return trimmed, ids
        trimmed.pop(remove_idx)


def compute_gsm8k_variant_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("correct"))
    accuracy = float(correct) / float(total) if total else 0.0

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for item in results:
        name = item.get("dataset") or "unknown"
        bucket = by_dataset.setdefault(name, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if item.get("correct"):
            bucket["correct"] += 1

    for bucket in by_dataset.values():
        total_bucket = bucket.get("total", 0)
        bucket["accuracy"] = (
            float(bucket["correct"]) / float(total_bucket) if total_bucket else 0.0
        )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_dataset": by_dataset,
    }
