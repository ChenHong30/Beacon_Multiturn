import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute beacon compression rate (token reduction) for Multi-IF-style logs."
    )
    parser.add_argument("--log", required=True, help="Path to the evaluation log (.json or .jsonl).")
    parser.add_argument(
        "--tokenizer-model",
        default="/data/hkustgz/model_weight/8_beacon_0_sink_distill_v2",
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
    args = parser.parse_args()

    log_path = Path(args.log)
    data = _read_json_or_jsonl(log_path)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    num_beacons = _safe_int(meta.get("num_beacons_per_segment"), 16)
    num_sinks = _safe_int(meta.get("num_sinks"), 4)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )

    counts = compute_compression(
        data=data,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
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
