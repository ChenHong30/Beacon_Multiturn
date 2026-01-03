import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant, you should strictly follow every instruction given by the user."
)
DEFAULT_TOKENIZER_MODEL = "/data/hkustgz/model_weight/16_beacon_4_sink"


@dataclass
class Counts:
    total_turns: int = 0
    total_samples: int = 0
    orig_prompt_tokens: int = 0
    comp_prompt_tokens: int = 0
    orig_history_tokens: int = 0
    comp_history_tokens: int = 0
    sum_turn_compression_percent: float = 0.0


def _read_json_or_jsonl(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        raise ValueError(f"Empty file: {path}")

    if stripped[0] in "{[":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
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


def _get_turn_user(turn: Dict[str, Any]) -> str:
    return str(turn.get("user") or turn.get("human") or turn.get("prompt") or "")


def _get_turn_bot(turn: Dict[str, Any]) -> str:
    return str(turn.get("bot") or turn.get("assistant") or turn.get("response") or "")


def _chat_chunk(tokenizer, role: str, content: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": role, "content": content}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _gen_prompt_tokens(
    tokenizer, system_prompt: Optional[str], enable_thinking: bool
) -> int:
    if system_prompt:
        dummy = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "x"},
        ]
    else:
        dummy = [{"role": "user", "content": "x"}]
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


def _detect_log_format(data: Dict[str, Any]) -> str:
    results = data.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            if "turns" in item:
                return "multi_if"
            if "history" in item and "question" in item:
                return "gsm8k_interference"
            if "turn_id" in item and (
                "dialogue_index" in item or "multi_id" in item or "id" in item
            ):
                return "mtbench_101"
    return "multi_if"


def _normalize_role(role: Optional[str]) -> str:
    role = (role or "user").lower()
    if role not in {"user", "assistant", "system"}:
        return "user"
    return role


def compute_gsm8k_compression(
    *,
    data: Dict[str, Any],
    tokenizer,
    system_prompt: str,
    answer_instruction: str,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
) -> Counts:
    per_history_message_kept = num_beacons + num_sinks
    system_tokens = (
        _count_tokens(
            tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
        )
        if system_prompt
        else 0
    )
    gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

    counts = Counts()

    for sample in _iter_samples(data):
        history = sample.get("history") or []
        question = str(sample.get("question") or "")
        if answer_instruction:
            question = f"{question}\n\n{answer_instruction}"

        if not question:
            continue

        counts.total_samples += 1
        counts.total_turns += 1

        history_tokens = 0
        history_messages = 0
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = _normalize_role(msg.get("role"))
            content = str(msg.get("content") or "")
            if not content:
                continue
            history_tokens += _count_tokens(
                tokenizer, _chat_chunk(tokenizer, role, content, enable_thinking)
            )
            history_messages += 1

        user_tokens = _count_tokens(
            tokenizer, _chat_chunk(tokenizer, "user", question, enable_thinking)
        )

        orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
        comp_prompt = (
            system_tokens + (history_messages * per_history_message_kept) + user_tokens + gen_prompt_tokens
        )

        counts.orig_prompt_tokens += orig_prompt
        counts.comp_prompt_tokens += comp_prompt
        counts.orig_history_tokens += history_tokens
        counts.comp_history_tokens += history_messages * per_history_message_kept

        if orig_prompt > 0:
            turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(orig_prompt)
            counts.sum_turn_compression_percent += turn_compression_percent

    return counts


def _default_mtbench_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "eval" / "mtbench_101" / "mtbench101.jsonl"


def _load_mtbench_conversations(path: Path) -> List[Dict[str, Any]]:
    data = _read_json_or_jsonl(path)
    conversations = list(_iter_samples(data))
    if not conversations:
        raise ValueError(f"No conversations found in mtbench data: {path}")
    return conversations


def _index_mtbench_dialogues(conversations: List[Dict[str, Any]]) -> Dict[Any, int]:
    id_to_index: Dict[Any, int] = {}
    for idx, dialogue in enumerate(conversations):
        if not isinstance(dialogue, dict):
            continue
        multi_id = dialogue.get("id")
        if multi_id is None:
            continue
        id_to_index[multi_id] = idx
        id_to_index[str(multi_id)] = idx
    return id_to_index


def _collect_mtbench_turns(
    results: Iterable[Dict[str, Any]],
    id_to_index: Dict[Any, int],
) -> Dict[int, Set[int]]:
    by_dialogue: Dict[int, Set[int]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        turn_id = item.get("turn_id")
        if turn_id is None:
            continue
        try:
            turn_index = int(turn_id) - 1
        except (TypeError, ValueError):
            continue
        if turn_index < 0:
            continue

        dialogue_index = item.get("dialogue_index")
        if dialogue_index is None:
            multi_id = item.get("multi_id") or item.get("id")
            if multi_id is not None:
                dialogue_index = id_to_index.get(multi_id)
                if dialogue_index is None:
                    dialogue_index = id_to_index.get(str(multi_id))
        try:
            dialogue_index = int(dialogue_index)
        except (TypeError, ValueError):
            continue

        by_dialogue.setdefault(dialogue_index, set()).add(turn_index)
    return by_dialogue


def compute_compression(
    *,
    data: Dict[str, Any],
    tokenizer,
    system_prompt: str,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
) -> Counts:
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


def compute_mtbench_compression(
    *,
    results: Iterable[Dict[str, Any]],
    conversations: List[Dict[str, Any]],
    tokenizer,
    num_beacons: int,
    num_sinks: int,
    enable_thinking: bool,
    override_system_prompt: Optional[str],
) -> Counts:
    per_history_message_kept = num_beacons + num_sinks

    id_to_index = _index_mtbench_dialogues(conversations)
    dialogue_turns = _collect_mtbench_turns(results, id_to_index)

    counts = Counts()

    for dialogue_index, turn_indices in dialogue_turns.items():
        if dialogue_index < 0 or dialogue_index >= len(conversations):
            continue
        if not turn_indices:
            continue
        dialogue = conversations[dialogue_index]
        turns = dialogue.get("history") or []
        if not isinstance(turns, list) or not turns:
            continue

        counts.total_samples += 1

        if override_system_prompt is not None:
            system_prompt = override_system_prompt
        else:
            system_prompt = str(dialogue.get("system") or dialogue.get("system_prompt") or "")

        system_tokens = (
            _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "system", system_prompt, enable_thinking)
            )
            if system_prompt
            else 0
        )
        gen_prompt_tokens = _gen_prompt_tokens(tokenizer, system_prompt, enable_thinking)

        history_tokens = 0
        history_messages = 0

        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            user = _get_turn_user(turn)
            user_tokens = _count_tokens(
                tokenizer, _chat_chunk(tokenizer, "user", user, enable_thinking)
            )
            bot = _get_turn_bot(turn)
            bot_tokens = (
                _count_tokens(
                    tokenizer, _chat_chunk(tokenizer, "assistant", bot, enable_thinking)
                )
                if bot
                else 0
            )

            if turn_index in turn_indices:
                orig_prompt = system_tokens + history_tokens + user_tokens + gen_prompt_tokens
                comp_prompt = (
                    system_tokens
                    + (history_messages * per_history_message_kept)
                    + user_tokens
                    + gen_prompt_tokens
                )

                counts.total_turns += 1
                counts.orig_prompt_tokens += orig_prompt
                counts.comp_prompt_tokens += comp_prompt
                counts.orig_history_tokens += history_tokens
                counts.comp_history_tokens += history_messages * per_history_message_kept

                if orig_prompt > 0:
                    turn_compression_percent = 100.0 * (orig_prompt - comp_prompt) / float(
                        orig_prompt
                    )
                    counts.sum_turn_compression_percent += turn_compression_percent

            history_tokens += user_tokens
            history_messages += 1
            if bot:
                history_tokens += bot_tokens
                history_messages += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute beacon compression rate (token reduction) for Multi-IF, MTBench_101, or GSM8K interference logs."
    )
    parser.add_argument("--log", required=True, help="Path to the evaluation log (.json or .jsonl).")
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help=(
            "Tokenizer/model path for token counting. If omitted, use log meta model_path, "
            "otherwise fallback to the default."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help=(
            "System prompt for Multi-IF logs (default: built-in). "
            "For MTBench_101, only used when explicitly provided to override dataset prompts."
        ),
    )
    parser.add_argument(
        "--mtbench-data",
        default=None,
        help=(
            "Path to mtbench101.jsonl when computing MTBench_101 logs "
            "(default: repo eval/mtbench_101/mtbench101.jsonl)."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to apply_chat_template (default: False).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    data = _read_json_or_jsonl(log_path)
    log_format = _detect_log_format(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    num_beacons = _safe_int(meta.get("num_beacons_per_segment"), 16)
    num_sinks = _safe_int(meta.get("num_sinks"), 4)

    tokenizer_model = (
        args.tokenizer_model
        or (meta.get("model_path") if isinstance(meta.get("model_path"), str) else None)
        or DEFAULT_TOKENIZER_MODEL
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )

    if log_format == "mtbench_101":
        if "num_beacons_per_segment" not in meta or "num_sinks" not in meta:
            raise ValueError(
                "MTBench_101 compression needs beacon logs with num_beacons_per_segment/num_sinks."
            )
        mtbench_path = Path(args.mtbench_data) if args.mtbench_data else _default_mtbench_data_path()
        if not mtbench_path.exists():
            raise FileNotFoundError(f"MTBench data file not found: {mtbench_path}")
        conversations = _load_mtbench_conversations(mtbench_path)
        counts = compute_mtbench_compression(
            results=_iter_samples(data),
            conversations=conversations,
            tokenizer=tokenizer,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
            override_system_prompt=args.system_prompt,
        )
    elif log_format == "gsm8k_interference":
        system_prompt = args.system_prompt
        if system_prompt is None:
            system_prompt = str(meta.get("system_prompt") or "")
        answer_instruction = str(meta.get("answer_instruction") or "")
        counts = compute_gsm8k_compression(
            data=data,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
        )
    else:
        system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
        counts = compute_compression(
            data=data,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            num_beacons=num_beacons,
            num_sinks=num_sinks,
            enable_thinking=bool(args.enable_thinking),
        )

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

    print(f"log: {log_path}")
    print(f"log_format: {log_format}")
    print(f"tokenizer_model: {tokenizer_model}")
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
