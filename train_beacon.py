import argparse
import inspect
import json
import logging
import math
import os
import shlex
import sys
from typing import Any, Dict, List, Optional, TextIO

import numpy as np

import math

import torch
from datasets import DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from modeling_qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


ALLOWED_ROLES = {"system", "user", "assistant"}


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(value) for value in obj]
    return str(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 beacon projections on multi-turn chat data.")
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs='+',  # 接受多个参数
        default=["lmsys-chat-turn-ge-3.jsonl"],
        help="Paths to JSONL files that store multi-turn conversations. Can specify multiple files separated by spaces.",
    )
    # 保持旧的兼容性
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the JSONL file that stores multi-turn conversations. (Deprecated: Use --data-paths instead)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Qwen2-0.5B-Instruct",
        help="Checkpoint directory used to initialise the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./beacon-finetune",
        help="Directory where the fine-tuned checkpoint will be written.",
    )
    parser.add_argument(
        "--processed-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache the processed tokenised dataset. "
        "If指定后，二次运行会直接从磁盘加载。",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt prepended to every conversation turn.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum number of tokens per packed sample after tokenisation.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.02,
        help="Portion of examples reserved for evaluation (0 disables evaluation).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied to all trainable parameters.",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Linear warmup ratio.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency of loss logging.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Frequency of checkpoint saving.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints retained on disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 mixed precision (preferred on Ampere+ GPUs).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision.",
    )
    parser.add_argument(
        "--train-beacon-only",
        action="store_true",
        help="Freeze all parameters except beacon projections (and beacon embedding).",
    )
    parser.add_argument(
        "--train-lm-head",
        action="store_true",
        help="When --train-beacon-only is set, also finetune the LM head.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to lower activation memory usage.",
    )
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        parser.error("Choose at most one of --fp16 or --bf16.")

    return args


def normalise_role(role: Optional[str]) -> str:
    if role is None:
        return "user"
    role = role.lower()
    if role not in ALLOWED_ROLES:
        if role in {"assistant-prefill", "gpt"}:
            return "assistant"
        return "user"
    return role


def build_messages(turns: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn in turns:
        role = normalise_role(turn.get("role"))
        content = turn.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        messages.append({"role": role, "content": content})
    return messages


def prepare_dataset(
    data_paths: List[str],
    tokenizer,
    max_length: int,
    system_prompt: str,
    eval_ratio: float,
    seed: int,
    processed_cache_dir: Optional[str],
):
    if processed_cache_dir and os.path.isdir(processed_cache_dir):
        logger.info("Loading processed dataset from %s", processed_cache_dir)
        cached_dataset = load_from_disk(processed_cache_dir)
        if isinstance(cached_dataset, DatasetDict):
            train_dataset = cached_dataset["train"]
            eval_dataset = cached_dataset.get("eval")
        else:
            train_dataset = cached_dataset
            eval_dataset = None
        return train_dataset, eval_dataset

    # 加载多个数据集并连接
    logger.info("Loading %d dataset files", len(data_paths))
    all_conversations = []
    
    for data_path in data_paths:
        logger.info("Loading dataset from %s", data_path)
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info("Loaded %d raw conversations from %s", len(dataset), data_path)
        
        # 直接提取conversation字段的值，跳过schema问题
        for example in dataset:
            # 标准化对话数据
            standard_conv = []
            for turn in example["conversation"]:
                standard_turn = {
                    "role": turn.get("role", ""),
                    "content": turn.get("content", "")
                }
                standard_conv.append(standard_turn)
            all_conversations.append({"conversation": standard_conv})
    
    # 创建标准格式的新数据集
    from datasets import Dataset
    dataset = Dataset.from_list(all_conversations)
    logger.info("Combined dataset has %d raw conversations", len(dataset))

    newline_token_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    newline_token_id = newline_token_ids[-1] if newline_token_ids else None
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        input_ids_batch: List[List[int]] = []
        attention_masks_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for turns in batch["conversation"]:
            if not isinstance(turns, list) or len(turns) == 0:
                continue

            messages = build_messages(turns, system_prompt)
            expanded_ids: List[int] = []
            expanded_labels: List[int] = []

            for message in messages:
                role = message["role"]
                content = message["content"]
                chunk = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                tokenised = tokenizer(chunk, add_special_tokens=False)
                chunk_ids = tokenised["input_ids"]
                if len(chunk_ids) == 0:
                    continue

                chunk_labels = [-100] * len(chunk_ids)
                if role == "assistant":
                    chunk_labels = chunk_ids.copy()

                    # mask prefix tokens (<|im_start|>, role, leading newline)
                    prefix_end = 0
                    if newline_token_id is not None:
                        for idx, token_id in enumerate(chunk_ids):
                            if token_id == newline_token_id:
                                prefix_end = idx
                                break
                        else:
                            prefix_end = min(2, len(chunk_ids) - 1)
                    else:
                        prefix_end = min(2, len(chunk_ids) - 1)
                    for idx in range(prefix_end + 1):
                        chunk_labels[idx] = -100

                    # mask trailing special tokens (<|im_end|> and trailing newline)
                    for idx in range(len(chunk_ids) - 2, len(chunk_ids)):
                        if idx >= 0:
                            chunk_labels[idx] = -100

                    # fallback to ensure <|im_start|> and <|im_end|> always masked
                    if im_start_id is not None and chunk_ids[0] == im_start_id:
                        chunk_labels[0] = -100
                    if im_end_id is not None and chunk_ids[-2] == im_end_id:
                        chunk_labels[-2] = -100

                expanded_ids.extend(chunk_ids)
                expanded_labels.extend(chunk_labels)

            if len(expanded_ids) == 0:
                continue

            attention_mask = [1] * len(expanded_ids)

            if len(expanded_ids) > max_length:
                expanded_ids = expanded_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                expanded_labels = expanded_labels[:max_length]

            if not any(label != -100 for label in expanded_labels):
                continue

            input_ids_batch.append(expanded_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(expanded_labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }

    processed = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenising conversations",
    )

    processed = processed.filter(lambda example: len(example["input_ids"]) > 1)

    if eval_ratio > 0.0:
        split = processed.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(
            "Dataset split into %d train and %d eval examples",
            len(train_dataset),
            len(eval_dataset),
        )
    else:
        train_dataset = processed
        eval_dataset = None
        logger.info("Using all %d examples for training (no eval split)", len(train_dataset))

    if processed_cache_dir:
        os.makedirs(processed_cache_dir, exist_ok=True)
        dataset_dict = DatasetDict({"train": train_dataset})
        if eval_dataset is not None:
            dataset_dict["eval"] = eval_dataset
        dataset_dict.save_to_disk(processed_cache_dir)
        logger.info("Saved processed dataset to %s", processed_cache_dir)

    return train_dataset, eval_dataset


def freeze_non_beacon_parameters(model: Qwen2ForCausalLM, train_lm_head: bool) -> None:
    beacon_keywords = {"beacon_q_proj", "beacon_k_proj", "beacon_v_proj"}
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        keep_trainable = any(keyword in name for keyword in beacon_keywords)

        if not keep_trainable and "embed_tokens" in name:
            keep_trainable = True  # allow beacon embedding to update
        if not keep_trainable and train_lm_head and "lm_head" in name:
            keep_trainable = True

        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable += param.numel()
        else:
            frozen += param.numel()

    logger.info(
        "Trainable params: %.2fM | Frozen params: %.2fM",
        trainable / 1e6,
        frozen / 1e6,
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    if predictions is None:
        return {}
    preds = np.argmax(predictions, axis=-1)
    mask = labels != -100
    valid_tokens = mask.sum()
    if valid_tokens == 0:
        token_accuracy = 0.0
    else:
        correct = (preds == labels) & mask
        token_accuracy = correct.sum() / valid_tokens
    return {"token_accuracy": float(token_accuracy)}


class Tee:
    """
    Mirrors writes to both the original stream and a log file, keeping console output while recording it.
    """

    def __init__(self, original: TextIO, log_file: TextIO):
        self.original = original
        self.log_file = log_file

    def write(self, data: str) -> None:
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()


class JsonMetricsCallback(TrainerCallback):
    """
    Records training metrics into a JSON file every time the Trainer logs.
    """

    def __init__(self, json_path: str, params: Dict[str, Any]):
        self.json_path = json_path
        self.data: Dict[str, Any] = {
            "params": params,
            "metrics": [],
        }

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs) -> None:
        if not logs or "loss" not in logs:
            return

        record: Dict[str, Any] = {
            "step": state.global_step,
            "epoch": state.epoch,
        }
        for key in ("loss", "learning_rate", "grad_norm"):
            if key in logs:
                record[key] = logs[key]
        if record:
            self.data["metrics"].append(record)
            self._write()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        self._write()

    def _write(self) -> None:
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as json_file:
            json.dump(self.data, json_file, ensure_ascii=False, indent=2)


def generate_loss_plot(metrics_json_path: str, plot_path: str) -> None:
    if not os.path.exists(metrics_json_path):
        logger.warning("Metrics JSON %s not found. Skipping loss plot generation.", metrics_json_path)
        return

    try:
        with open(metrics_json_path, "r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read metrics JSON %s: %s", metrics_json_path, exc)
        return

    metrics = payload.get("metrics", [])
    steps = []
    losses = []
    for record in metrics:
        step = record.get("step")
        loss = record.get("loss")
        if loss is None:
            continue
        steps.append(step if step is not None else len(steps))
        losses.append(loss)

    if not losses:
        logger.warning("No loss entries found in %s. Skipping loss plot generation.", metrics_json_path)
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.warning("matplotlib is not installed. Skipping loss plot generation.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", linestyle="-", linewidth=1.5, markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved loss plot to %s", plot_path)


class BeaconDataCollator:
    """
    Pads variable-length chat examples while preserving precomputed labels.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for feature in features:
            input_ids = feature["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            attention_mask = feature.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()

            labels = feature.get("labels")
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            pad_len = max_length - len(input_ids)

            padded_input_ids = input_ids + [pad_token_id] * pad_len
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            padded_attention_mask = attention_mask + [0] * pad_len

            if labels is None:
                labels = padded_input_ids.copy()
            padded_labels = labels + [-100] * pad_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

def main() -> None:
    args = parse_args()

    # 处理新旧参数兼容性
    if args.data_path is not None:
        # 如果使用了旧参数，则只使用旧参数
        data_paths = [args.data_path]
        logger.warning("Using deprecated --data-path argument. Consider using --data-paths instead.")
    else:
        # 否则使用新参数
        data_paths = args.data_paths

    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    log_path = os.path.join(logging_dir, "training_logs.log")
    metrics_json_path = os.path.join(logging_dir, "training_metrics.json")
    loss_plot_path = os.path.join(logging_dir, "loss_curve.png")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file: Optional[TextIO] = None
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        command = " ".join(shlex.quote(arg) for arg in sys.argv)
        log_file.write(f"[command] {command}\n")
        log_file.write(f"[cwd] {os.getcwd()}\n")
        log_file.write("\n")
        log_file.flush()

        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            level=logging.INFO,
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_dataset, eval_dataset = prepare_dataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            processed_cache_dir=args.processed_cache_dir,
        )

        model = Qwen2ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.config.use_cache = False

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if args.train_beacon_only:
            freeze_non_beacon_parameters(model, train_lm_head=args.train_lm_head)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "Training all parameters (%.2fM trainable of %.2fM total)",
                trainable_params / 1e6,
                total_params / 1e6,
            )

        data_collator = BeaconDataCollator(tokenizer=tokenizer)

        evaluation_strategy = "steps" if eval_dataset is not None else "no"
        training_kwargs = {
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "evaluation_strategy": evaluation_strategy,
            "logging_strategy": "steps",
            "logging_dir": logging_dir,
            "report_to": "none",
            "dataloader_pin_memory": True,
            "dataloader_drop_last": False,
            "gradient_checkpointing": args.gradient_checkpointing,
            "remove_unused_columns": False,
            "label_names": ["labels"],
            "optim": "adamw_torch",
            "seed": args.seed,
            "prediction_loss_only": False,
        }
        if eval_dataset is not None:
            training_kwargs["eval_steps"] = args.save_steps

        supported_kwargs = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        unsupported = [key for key in list(training_kwargs.keys()) if key not in supported_kwargs]
        for key in unsupported:
            logger.warning("Dropping unsupported TrainingArguments argument: %s", key)
            training_kwargs.pop(key)

        training_args = TrainingArguments(**training_kwargs)

        params_for_json = {
            "cli_args": sanitize_for_json(vars(args)),
            "training_arguments": sanitize_for_json(training_args.to_dict()),
        }
        metrics_callback = JsonMetricsCallback(metrics_json_path, params_for_json)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
            callbacks=[metrics_callback],
        )

        trainer.train()

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Training complete. Checkpoint saved to %s", args.output_dir)
        generate_loss_plot(metrics_json_path, loss_plot_path)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file is not None:
            log_file.flush()
            log_file.close()


if __name__ == "__main__":
    main()
