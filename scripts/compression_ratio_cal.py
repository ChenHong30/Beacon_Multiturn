import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant, you should strictly follow every instruction given by the user."
)


@dataclass
class Counts:
    total_turns: int = 0
    total_samples: int = 0
    orig_prompt_tokens: int = 0
    comp_prompt_tokens: int = 0
    orig_history_tokens: int = 0
    comp_history_tokens: int = 0
    sum_turn_compression_percent: float = 0.0


def detect_task_type(data: Dict[str, Any]) -> str:
    """Automatically detect task type from log data structure."""
    results = data.get("results", [])
    if not results or not isinstance(results, list) or len(results) == 0:
        raise ValueError("No results found in log file")

    sample = results[0]

    # Check for safediabench
    if "identification_score" in sample or "handling_score" in sample:
        return "safediabench"

    # Check for mhj: has "turns" with "user_content" and "response", plus "jailbroken" field
    if "turns" in sample and isinstance(sample["turns"], list):
        if len(sample["turns"]) > 0 and "user_content" in sample["turns"][0]:
            return "mhj"
        # Check for multi_if: has "turns" field with "prompt" and "response"
        if len(sample["turns"]) > 0 and "prompt" in sample["turns"][0]:
            return "multi_if"

    # Check for mtbench_101: has "task" and "multi_id" fields
    if "task" in sample and "multi_id" in sample and "turn_id" in sample:
        return "mtbench_101"

    # Check for gsm8k_variant or coreference_resolution: has "conversation" field
    if "conversation" in sample and isinstance(sample["conversation"], list):
        dataset = sample.get("dataset", "")
        if "gsm8k" in dataset:
            return "gsm8k_variant"
        elif "coref" in dataset:
            return "coreference_resolution"

    raise ValueError(f"Unable to detect task type from log structure. Sample keys: {list(sample.keys())}")


def _read_json_or_jsonl(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        raise ValueError(f"Empty file: {path}")

    if stripped[0] in "{[":
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"results": data, "meta": {}}
        raise TypeError(f"Unsupported JSON root type: {type(data)}")

    results: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        results.append(json.loads(line))
    return {"results": results, "meta": {}}


def _iter_samples(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    results = data.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                yield item


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chat_chunk(tokenizer, role: str, content: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": role, "content": content}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _gen_prompt_tokens(tokenizer, system_prompt: str, enable_thinking: bool) -> int:
    dummy = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "x"},
    ]
    no_gen = tokenizer.apply_chat_template(
        dummy,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    with_gen = tokenizer.apply_chat_template(
        dummy,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return _count_tokens(tokenizer, with_gen) - _count_tokens(tokenizer, no_gen)


def compute_compression_multi_if(
    *,
    data: Dict[str, Any],
    tokenizer,
    system_prompt: str,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
) -> Counts:
    """Compute compression for Multi-IF task (original implementation)."""
    per_history_message_kept = num_beacons + num_sinks

    system_tokens = _count_tokens(
        tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
    )
    gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

    counts = Counts()

    for sample in _iter_samples(data):
        turns = sample.get("turns") or []
        if not isinstance(turns, list) or not turns:
            continue

        counts.total_samples += 1

        history_tokens = 0
        history_messages = 0

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            prompt = str(turn.get("prompt") or "")
            response = str(turn.get("response") or "")

            user_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "user", prompt, enable_thinking)
            )
            assistant_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "assistant", response, enable_thinking)
            )

            orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
            comp_prompt = (
                system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
            )

            counts.total_turns += 1
            counts.orig_prompt_tokens += orig_prompt
            counts.comp_prompt_tokens += comp_prompt
            counts.orig_history_tokens += history_tokens
            counts.comp_history_tokens += history_messages * per_history_message_kept

            if orig_prompt > 0:
                turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
                counts.sum_turn_compression_percent += turn_compression_percent

            history_tokens += user_tokens + assistant_tokens
            history_messages += 2

    return counts


def compute_compression_conversation_based(
    *,
    data: Dict[str, Any],
    tokenizer,
    system_prompt: str,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
) -> Counts:
    """Compute compression for conversation-based tasks (gsm8k_variant, coreference_resolution)."""
    per_history_message_kept = num_beacons + num_sinks

    system_tokens = _count_tokens(
        tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
    )
    gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

    counts = Counts()

    for sample in _iter_samples(data):
        conversation = sample.get("conversation") or []
        if not isinstance(conversation, list) or not conversation:
            continue

        counts.total_samples += 1

        history_tokens = 0
        history_messages = 0

        # Process conversation turn by turn
        for i, msg in enumerate(conversation):
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "user":
                # Calculate compression at each user turn (before response)
                user_tokens = _count_tokens(
                    tokenizer, _chat_chunk(tokenizer, "user", content, enable_thinking)
                )

                orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
                comp_prompt = (
                    system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
                )

                counts.total_turns += 1
                counts.orig_prompt_tokens += orig_prompt
                counts.comp_prompt_tokens += comp_prompt
                counts.orig_history_tokens += history_tokens
                counts.comp_history_tokens += history_messages * per_history_message_kept

                if orig_prompt > 0:
                    turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
                    counts.sum_turn_compression_percent += turn_compression_percent

                history_tokens += user_tokens
                history_messages += 1

            elif role == "assistant":
                # Add assistant response to history
                assistant_tokens = _count_tokens(
                    tokenizer, _chat_chunk(tokenizer, "assistant", content, enable_thinking)
                )
                history_tokens += assistant_tokens
                history_messages += 1

    return counts


def compute_compression_mhj(
    *,
    data: Dict[str, Any],
    tokenizer,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
    dataset_path: Optional[str] = None,
) -> Counts:
    """
    Compute compression for MHJ (Multi-turn Human Jailbreaking) task.

    MHJ log structure:
    - results: list of dialogues
      - Each dialogue has: id, source, tactic, turns (list)
      - Each turn has: user_content, response, judge_result

    We need to load the original dataset to get system prompts.
    """
    per_history_message_kept = num_beacons + num_sinks

    # Load original dataset to get system prompts
    if not dataset_path:
        script_dir = Path(__file__).parent.parent
        dataset_path = script_dir / "eval" / "mhj" / "mhj_dataset.jsonl"

    # Build mapping from id to system prompt
    id_to_system: Dict[str, str] = {}
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = str(item.get("id", ""))
                system_prompt = item.get("system", DEFAULT_SYSTEM_PROMPT)
                id_to_system[item_id] = system_prompt

    counts = Counts()

    for sample in _iter_samples(data):
        dialogue_id = str(sample.get("id", ""))
        turns = sample.get("turns") or []
        if not isinstance(turns, list) or not turns:
            continue

        counts.total_samples += 1

        # Get system prompt for this dialogue
        system_prompt = id_to_system.get(dialogue_id, DEFAULT_SYSTEM_PROMPT)
        system_tokens = _count_tokens(
            tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
        )
        gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

        history_tokens = 0
        history_messages = 0

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            user_content = str(turn.get("user_content") or "")
            response = str(turn.get("response") or "")

            user_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "user", user_content, enable_thinking)
            )
            assistant_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "assistant", response, enable_thinking)
            )

            orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
            comp_prompt = (
                system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
            )

            counts.total_turns += 1
            counts.orig_prompt_tokens += orig_prompt
            counts.comp_prompt_tokens += comp_prompt
            counts.orig_history_tokens += history_tokens
            counts.comp_history_tokens += history_messages * per_history_message_kept

            if orig_prompt > 0:
                turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
                counts.sum_turn_compression_percent += turn_compression_percent

            history_tokens += user_tokens + assistant_tokens
            history_messages += 2

    return counts


def compute_compression_safediabench(
    *,
    data: Dict[str, Any],
    tokenizer,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
    dataset_path: Optional[str] = None,
) -> Counts:
    """
    Compute compression for SafeDialBench task.
    """
    per_history_message_kept = num_beacons + num_sinks

    # Load original dataset to get system prompts and history
    if not dataset_path:
        script_dir = Path(__file__).parent.parent
        dataset_path = script_dir / "eval" / "safediabench" / "safediabench_dataset.jsonl"

    # Build mapping from id to dialogue
    id_to_dialogue: Dict[str, Any] = {}
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_id = str(item.get("id", ""))
                id_to_dialogue[item_id] = item

    counts = Counts()

    for sample in _iter_samples(data):
        dialogue_id = str(sample.get("id", ""))
        turns = sample.get("turns") or []
        if not isinstance(turns, list) or not turns:
            continue
        
        # Get original dialogue
        orig_dialogue = id_to_dialogue.get(dialogue_id)
        if not orig_dialogue:
             continue

        counts.total_samples += 1

        # System prompt
        system_prompt = orig_dialogue.get("system")
        system_tokens = 0
        gen_prompt_tokens = 0
        
        # Calculate system tokens and gen prompt overhead
        if system_prompt:
             system_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
             )
             gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)
        else:
             # Calculate generation overhead without system prompt
             dummy = [{"role": "user", "content": "x"}]
             no_gen = tokenizer.apply_chat_template(
                dummy, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
             )
             with_gen = tokenizer.apply_chat_template(
                dummy, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
             )
             gen_prompt_tokens = _count_tokens(tokenizer, with_gen) - _count_tokens(tokenizer, no_gen)

        history_tokens = 0
        history_messages = 0
        
        # Original history from dataset
        orig_history = orig_dialogue.get("history", [])

        # Process each turn in the log
        for i, turn_log in enumerate(turns):
            if i >= len(orig_history):
                break
                
            # Current turn user content
            user_content = orig_history[i].get("user", "")
            
            user_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "user", user_content, enable_thinking)
            )
            
            orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
            comp_prompt = (
                system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
            )

            counts.total_turns += 1
            counts.orig_prompt_tokens += orig_prompt
            counts.comp_prompt_tokens += comp_prompt
            counts.orig_history_tokens += history_tokens
            counts.comp_history_tokens += history_messages * per_history_message_kept

            if orig_prompt > 0:
                turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
                counts.sum_turn_compression_percent += turn_compression_percent

            # Update history with GROUND TRUTH response for next turn
            bot_content = orig_history[i].get("bot", "")
            assistant_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "assistant", bot_content, enable_thinking)
            )
            
            history_tokens += user_tokens + assistant_tokens
            history_messages += 2

    return counts


def compute_compression_mtbench(
    *,
    data: Dict[str, Any],
    tokenizer,
    system_prompt: str,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
    dataset_path: Optional[str] = None,
) -> Counts:
    """Compute compression for MTBench-101 task."""
    per_history_message_kept = num_beacons + num_sinks

    system_tokens = _count_tokens(
        tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
    )
    gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

    # Load original dataset to get dialogue history
    if not dataset_path:
        # Try to find dataset in default location
        script_dir = Path(__file__).parent.parent
        dataset_path = script_dir / "eval" / "mtbench_101" / "mtbench101.jsonl"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"MTBench-101 dataset not found at {dataset_path}. Please specify --dataset-path.")

    # Load dialogues from dataset
    dialogues = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            dialogue = json.loads(line)
            dialogues[idx] = dialogue

    counts = Counts()
    processed_dialogues = set()

    for sample in _iter_samples(data):
        dialogue_index = sample.get("dialogue_index")
        turn_id = int(str(sample.get("turn_id", "1")))

        if dialogue_index is None:
            continue

        if dialogue_index not in dialogues:
            continue

        dialogue = dialogues[dialogue_index]
        history = dialogue.get("history", [])

        # Skip first tasks based on task type
        task = sample.get("task", "")
        skip_first_tasks = ['FR', 'CR', 'AR', 'SA', 'SC', 'CM']
        skip_first = task in skip_first_tasks

        # Count samples (unique dialogues)
        dialogue_key = (dialogue_index, turn_id)
        if dialogue_index not in processed_dialogues:
            processed_dialogues.add(dialogue_index)
            counts.total_samples += 1

        # Calculate compression for this turn
        turn_index = turn_id - 1
        if turn_index < 0 or turn_index >= len(history):
            continue

        if skip_first and turn_index == 0:
            continue

        # Build history up to current turn
        history_tokens = 0
        history_messages = 0

        for i in range(turn_index):
            user_msg = str(history[i].get("user", ""))
            bot_msg = str(history[i].get("bot", ""))

            user_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "user", user_msg, enable_thinking)
            )
            assistant_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "assistant", bot_msg, enable_thinking)
            )

            history_tokens += user_tokens + assistant_tokens
            history_messages += 2

        # Current turn user message
        current_user = str(history[turn_index].get("user", ""))
        user_tokens = _count_tokens(
            tokenizer, _chat_chunk(tokenizer, "user", current_user, enable_thinking)
        )

        orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
        comp_prompt = (
            system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
        )

        counts.total_turns += 1
        counts.orig_prompt_tokens += orig_prompt
        counts.comp_prompt_tokens += comp_prompt
        counts.orig_history_tokens += history_tokens
        counts.comp_history_tokens += history_messages * per_history_message_kept

        if orig_prompt > 0:
            turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
            counts.sum_turn_compression_percent += turn_compression_percent

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute beacon compression rate (token reduction) for evaluation logs. "
                    "Automatically detects task type from log structure."
    )
    parser.add_argument("--log", required=True, help="Path to the evaluation log (.json or .jsonl).")
    parser.add_argument(
        "--tokenizer-model",
        default="/data/hkustgz/model_weight/Qwen3-0.6B",
        help="Tokenizer/model path for token counting.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when building the dialogue.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to apply_chat_template (default: False).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to original dataset (only needed for MTBench-101 if auto-detection fails).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    data = _read_json_or_jsonl(log_path)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    # Auto-detect task type
    task_type = detect_task_type(data)
    print(f"Detected task type: {task_type}")

    num_beacons = _safe_int(meta.get("num_beacons_per_segment"), 16)
    num_sinks = _safe_int(meta.get("num_sinks"), 4)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )

    # Compute compression based on task type
    if task_type == "multi_if":
        counts = compute_compression_multi_if(
            data=data,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
        )
    elif task_type == "mhj":
        counts = compute_compression_mhj(
            data=data,
            tokenizer=tokenizer,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
            dataset_path=args.dataset_path,
        )
    elif task_type == "safediabench":
        counts = compute_compression_safediabench(
            data=data,
            tokenizer=tokenizer,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
            dataset_path=args.dataset_path,
        )
    elif task_type == "mtbench_101":
        counts = compute_compression_mtbench(
            data=data,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
            dataset_path=args.dataset_path,
        )
    elif task_type in ["gsm8k_variant", "coreference_resolution"]:
        counts = compute_compression_conversation_based(
            data=data,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    def pct(reduced: int, original: int) -> float:
        if original <= 0:
            return 0.0
        return 100.0 * (1.0 - float(original - reduced) / float(original))

    # Full-prompt token reduction: 0% means no reduction, 100% means all tokens removed.
    prompt_retained_ratio = (
        float(counts.comp_prompt_tokens) / float(counts.orig_prompt_tokens)
        if counts.orig_prompt_tokens
        else 0.0
    )
    prompt_compression_percent = (
        100.0 * (1.0 - prompt_retained_ratio) if counts.orig_prompt_tokens else 0.0
    )

    history_retained_ratio = (
        float(counts.comp_history_tokens) / float(counts.orig_history_tokens)
        if counts.orig_history_tokens
        else 0.0
    )
    history_compression_percent = (
        100.0 * (1.0 - history_retained_ratio) if counts.orig_history_tokens else 0.0
    )

    mean_turn_compression_percent = (
        counts.sum_turn_compression_percent / float(counts.total_turns)
        if counts.total_turns
        else 0.0
    )

    print(f"task_type: {task_type}")
    print(f"log: {log_path}")
    print(f"tokenizer_model: {args.tokenizer_model}")
    print(f"num_samples: {counts.total_samples}")
    print(f"total_turns: {counts.total_turns}")
    print(f"num_beacons: {num_beacons}")
    print(f"num_sinks: {num_sinks}")
    print(f"per_history_message_kept_tokens: {num_beacons + num_sinks}")
    print("")
    print(f"orig_prompt_tokens_total: {counts.orig_prompt_tokens}")
    print(f"compressed_prompt_tokens_total: {counts.comp_prompt_tokens}")
    print(f"avg_prompt_retained_ratio: {prompt_retained_ratio:.6f}")
    print(f"avg_prompt_compression_percent: {prompt_compression_percent:.4f}")
    print(f"mean_turn_compression_percent: {mean_turn_compression_percent:.4f}")
    print("")
    print(f"orig_history_tokens_total: {counts.orig_history_tokens}")
    print(f"compressed_history_tokens_total: {counts.comp_history_tokens}")
    print(f"avg_history_retained_ratio: {history_retained_ratio:.6f}")
    print(f"avg_history_compression_percent: {history_compression_percent:.4f}")


if __name__ == "__main__":
    main()
