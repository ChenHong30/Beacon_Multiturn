import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


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


def load_gsm8k(path: Optional[str], split: str) -> List[Dict[str, Any]]:
    if path:
        if path.endswith(".jsonl"):
            data = read_jsonl(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, dict) and "question" in data:
            return [data]
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported GSM8K data format: {path}")

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not available. Provide --gsm8k-path to a local GSM8K file."
        )
    return list(load_dataset("gsm8k", "main", split=split))


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


def build_line_offsets(path: str) -> List[int]:
    offsets: List[int] = []
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            offsets.append(offset)
            offset += len(line)
    return offsets


def count_conversation_rounds(conversation: List[Dict[str, Any]]) -> int:
    rounds = 0
    prev_role: Optional[str] = None
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role"))
        if role == "system":
            continue
        if prev_role == "user" and role == "assistant":
            rounds += 1
        prev_role = role
    return rounds


def build_round_buckets(path: str) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = item.get("conversation") or []
            if not isinstance(conv, list):
                continue
            rounds = count_conversation_rounds(conv)
            if rounds <= 0:
                continue
            buckets.setdefault(rounds, []).append(idx)
    return buckets


def sample_history_indices(
    num_samples: int, num_histories: int, seed: int
) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(num_histories) for _ in range(num_samples)]


def load_histories(
    path: str, offsets: List[int], indices: List[int]
) -> List[List[Dict[str, Any]]]:
    histories: List[List[Dict[str, Any]]] = []
    with open(path, "rb") as f:
        for idx in indices:
            f.seek(offsets[idx])
            line = f.readline().decode("utf-8").strip()
            if not line:
                histories.append([])
                continue
            item = json.loads(line)
            histories.append(item.get("conversation") or [])
    return histories


def _pick_index_from_buckets(
    candidate_rounds: List[int],
    rng: random.Random,
    round_buckets: Dict[int, List[int]],
) -> Tuple[Optional[int], Optional[int]]:
    for rounds in candidate_rounds:
        indices = round_buckets.get(rounds) or []
        if indices:
            return rng.choice(indices), rounds
    return None, None


def _build_pool_indices(
    total_samples: int,
    target_rounds: int,
    seed: int,
    round_buckets: Dict[int, List[int]],
) -> List[int]:
    if total_samples <= 0:
        return []

    max_round = max(round_buckets) if round_buckets else 0
    if target_rounds > max_round:
        candidate_rounds = [max_round] if max_round > 0 else []
    else:
        candidate_rounds = list(range(target_rounds, max_round + 1))

    rng = random.Random(seed)
    pool: List[int] = []
    for rounds in candidate_rounds:
        indices = list(round_buckets.get(rounds) or [])
        rng.shuffle(indices)
        pool.extend(indices)

    if not pool:
        return []

    while len(pool) < total_samples:
        pool.append(rng.choice(pool))

    return pool[:total_samples]


def build_history_plans(
    total_samples: int,
    history_max_turns: Optional[int],
    seed: int,
    round_buckets: Dict[int, List[int]],
) -> List[List[Tuple[int, int]]]:
    if total_samples <= 0:
        return []
    if history_max_turns is None or history_max_turns <= 0:
        return [[] for _ in range(total_samples)]
    if not round_buckets:
        return [[] for _ in range(total_samples)]

    if history_max_turns <= 7:
        pool = _build_pool_indices(total_samples, history_max_turns, seed, round_buckets)
        if not pool:
            return [[] for _ in range(total_samples)]
        return [[(pool[i], history_max_turns)] for i in range(total_samples)]

    plans: List[List[Tuple[int, int]]] = []
    for sample_idx in range(total_samples):
        rng = random.Random(seed + sample_idx)
        remaining = history_max_turns
        segments: List[Tuple[int, int]] = []

        idx, rounds = _pick_index_from_buckets([7, 6, 5, 4, 3, 2, 1], rng, round_buckets)
        if idx is None or rounds is None:
            plans.append([])
            continue

        take = min(remaining, min(7, rounds))
        if take > 0:
            segments.append((idx, take))
            remaining -= take

        while remaining > 0:
            idx, rounds = _pick_index_from_buckets([6, 5, 4, 3, 2, 1], rng, round_buckets)
            if idx is None or rounds is None:
                break
            take = min(remaining, rounds)
            if take <= 0:
                break
            segments.append((idx, take))
            remaining -= take

        plans.append(segments)

    return plans


def slice_conversation_by_rounds(
    conversation: List[Dict[str, Any]],
    max_rounds: int,
) -> List[Dict[str, Any]]:
    if max_rounds <= 0:
        return []

    result: List[Dict[str, Any]] = []
    rounds = 0
    prev_role: Optional[str] = None

    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role"))
        if role == "system":
            continue
        content = msg.get("content")
        result.append({"role": role, "content": content})
        if prev_role == "user" and role == "assistant":
            rounds += 1
            if rounds >= max_rounds:
                break
        prev_role = role

    return result


def build_histories_from_plans(
    path: str,
    offsets: List[int],
    plans: List[List[Tuple[int, int]]],
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    unique_indices: List[int] = []
    seen = set()
    for plan in plans:
        for idx, _ in plan:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

    conv_map: Dict[int, List[Dict[str, Any]]] = {}
    if unique_indices:
        conversations = load_histories(path, offsets, unique_indices)
        conv_map = dict(zip(unique_indices, conversations))

    histories: List[List[Dict[str, Any]]] = []
    metas: List[Dict[str, Any]] = []
    for plan in plans:
        history: List[Dict[str, Any]] = []
        source_messages = 0
        source_turns = 0
        indices: List[int] = []
        rounds_used: List[int] = []
        for idx, rounds in plan:
            conv = conv_map.get(idx, [])
            indices.append(idx)
            rounds_used.append(rounds)
            source_messages += len(conv)
            source_turns += count_conversation_rounds(conv)
            history.extend(slice_conversation_by_rounds(conv, rounds))
        histories.append(history)
        metas.append(
            {
                "history_indices": indices,
                "history_rounds": rounds_used,
                "history_source_messages": source_messages,
                "history_source_turns": source_turns,
            }
        )
    return histories, metas


def trim_history_messages(
    messages: List[Dict[str, Any]],
    max_turns: Optional[int],
) -> List[Dict[str, Any]]:
    if max_turns is None:
        return messages
    if max_turns <= 0:
        return []
    max_messages = max_turns * 2
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


def truncate_to_max_tokens(
    messages: List[Dict[str, Any]],
    tokenizer,
    max_input_tokens: Optional[int],
    enable_thinking: bool,
) -> Tuple[List[Dict[str, Any]], List[int]]:
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


def prepare_messages(
    history: List[Dict[str, Any]],
    question: str,
    answer_instruction: str,
    system_prompt: str,
    history_max_turns: Optional[int],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    history = trim_history_messages(history, history_max_turns)
    for msg in history:
        role = normalize_role(msg.get("role"))
        content = str(msg.get("content") or "")
        if not content:
            continue
        messages.append({"role": role, "content": content})

    final_question = question.strip()
    if answer_instruction:
        final_question = f"{final_question}\n\n{answer_instruction}"
    messages.append({"role": "user", "content": final_question})
    return messages


def extract_history_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return []
    start = 1 if messages[0].get("role") == "system" else 0
    end = len(messages) - 1
    if end <= start:
        return []
    return [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in messages[start:end]
    ]


def count_history_turns(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg.get("role") == "user")


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def get_dtype(name: str) -> torch.dtype:
    name = (name or "auto").lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def compute_gsm8k_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("correct"))
    accuracy = float(correct) / float(total) if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
