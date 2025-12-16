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

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

from modeling_qwen3 import Qwen3ForCausalLM

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
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 with LoRA while also training beacon parameters on multi-turn chat data."
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        default=["lmsys-chat-turn-ge-3.jsonl"],
        help="Paths to JSONL files that store multi-turn conversations.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the JSONL file that stores multi-turn conversations. (Deprecated: Use --data-paths instead)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Qwen3-8B",
        help="Checkpoint directory used to initialise the model (can be a beacon-trained checkpoint).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./beacon-lora-finetune",
        help="Directory where the LoRA adapter, logs, and beacon weights will be written.",
    )
    parser.add_argument(
        "--processed-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache the processed tokenised dataset. If set, subsequent runs load from disk.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt prepended to every conversation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum number of tokens per packed sample after tokenisation.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.02,
        help="Portion of examples reserved for evaluation (0 disables evaluation).",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay applied to all trainable params.")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Linear warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Frequency of loss logging.")
    parser.add_argument("--save-steps", type=int, default=500, help="Frequency of checkpoint saving.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument(
        "--num-beacons",
        type=int,
        default=16,
        help="Number of beacon tokens inserted per historical segment.",
    )
    parser.add_argument(
        "--num-sinks",
        type=int,
        default=4,
        help="Number of sink tokens retained at the beginning of each turn.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder to resume training from.",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module name suffixes to apply LoRA to.",
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias training mode.",
    )

    # What non-LoRA base params to train
    parser.add_argument(
        "--train-lm-head",
        action="store_true",
        help="Also finetune lm_head in addition to beacon params and LoRA adapters.",
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

    logger.info("Loading %d dataset files", len(data_paths))
    all_conversations: List[Dict[str, Any]] = []
    for data_path in data_paths:
        logger.info("Loading dataset from %s", data_path)
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info("Loaded %d raw conversations from %s", len(dataset), data_path)
        for example in dataset:
            raw_conv = example.get("conversation")
            if not isinstance(raw_conv, list):
                continue
            standard_conv = []
            for turn in raw_conv:
                if not isinstance(turn, dict):
                    continue
                standard_conv.append(
                    {
                        "role": turn.get("role", ""),
                        "content": turn.get("content", ""),
                    }
                )
            if standard_conv:
                all_conversations.append({"conversation": standard_conv})

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

                    for idx in range(len(chunk_ids) - 2, len(chunk_ids)):
                        if idx >= 0:
                            chunk_labels[idx] = -100

                    if im_start_id is not None and chunk_ids and chunk_ids[0] == im_start_id:
                        chunk_labels[0] = -100
                    if im_end_id is not None and len(chunk_ids) >= 2 and chunk_ids[-2] == im_end_id:
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
        logger.info("Dataset split into %d train and %d eval examples", len(train_dataset), len(eval_dataset))
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


def freeze_non_beacon_parameters(model, train_lm_head: bool) -> None:
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        keep_trainable = "beacon" in name
        if not keep_trainable and train_lm_head and "lm_head" in name:
            keep_trainable = True
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable += param.numel()
        else:
            frozen += param.numel()

    logger.info("Trainable base params: %.2fM | Frozen base params: %.2fM", trainable / 1e6, frozen / 1e6)


def _count_trainable_params(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": total / 1e6, "trainable_m": trainable / 1e6}


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
    def __init__(self, json_path: str, params: Dict[str, Any]):
        self.json_path = json_path
        self.data: Dict[str, Any] = {"params": params, "metrics": []}

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs) -> None:
        if not logs or "loss" not in logs:
            return
        record: Dict[str, Any] = {"step": state.global_step, "epoch": state.epoch}
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


def _get_beacon_state_dict(model) -> Dict[str, torch.Tensor]:
    state = {}
    for name, tensor in model.state_dict().items():
        if "beacon" in name:
            state[name] = tensor.detach().cpu()
    return state


def _save_beacon_weights(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "beacon_weights.safetensors")
    meta_path = os.path.join(output_dir, "beacon_weights.meta.json")

    try:
        from safetensors.torch import save_file as safe_save_file  # type: ignore
    except Exception:
        safe_save_file = None

    state = _get_beacon_state_dict(model)
    if safe_save_file is not None:
        safe_save_file(state, weights_path)
    else:
        torch.save(state, weights_path + ".pt")

    meta = {
        "format": "beacon_only",
        "num_keys": len(state),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_beacon_weights_if_present(model, checkpoint_dir: str) -> None:
    weights_path = os.path.join(checkpoint_dir, "beacon_weights.safetensors")
    weights_pt_path = weights_path + ".pt"
    state = None

    if os.path.isfile(weights_path):
        try:
            from safetensors.torch import load_file as safe_load_file  # type: ignore

            state = safe_load_file(weights_path, device="cpu")
        except Exception as e:
            logger.warning("Failed to load %s: %s", weights_path, e)
    elif os.path.isfile(weights_pt_path):
        try:
            state = torch.load(weights_pt_path, map_location="cpu")
        except Exception as e:
            logger.warning("Failed to load %s: %s", weights_pt_path, e)

    if not state:
        return

    try:
        load_result = model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded beacon weights from %s (missing=%d unexpected=%d)",
            checkpoint_dir,
            len(getattr(load_result, "missing_keys", []) or []),
            len(getattr(load_result, "unexpected_keys", []) or []),
        )
    except Exception as e:
        logger.warning("Failed to apply beacon weights from %s: %s", checkpoint_dir, e)


class BeaconWeightsCallback(TrainerCallback):
    """
    Saves beacon-only weights into each checkpoint folder and the final output_dir.
    This complements PEFT adapter checkpoints so that resuming/serving keeps beacon updates.
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def on_save(self, args, state, control, **kwargs) -> None:
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            _save_beacon_weights(self.base_model, ckpt_dir)
        except Exception as e:
            logger.warning("Failed to save beacon weights to %s: %s", ckpt_dir, e)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        try:
            _save_beacon_weights(self.base_model, args.output_dir)
        except Exception as e:
            logger.warning("Failed to save beacon weights to %s: %s", args.output_dir, e)


class BeaconDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

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

    if args.data_path is not None:
        data_paths = [args.data_path]
        logger.warning("Using deprecated --data-path argument. Consider using --data-paths instead.")
    else:
        data_paths = args.data_paths

    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    log_path = os.path.join(logging_dir, "training_logs.log")
    metrics_json_path = os.path.join(logging_dir, "training_metrics.json")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file: Optional[TextIO] = None

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        command = " ".join(shlex.quote(arg) for arg in sys.argv)
        log_file.write(f"[command] {command}\n")
        log_file.write(f"[cwd] {os.getcwd()}\n\n")
        log_file.flush()

        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)

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

        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.num_beacons_per_segment = args.num_beacons
        config.num_sinks = args.num_sinks

        model = Qwen3ForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            tokenizer=tokenizer,
            dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            device_map=None,
            attn_implementation="eager",
        )
        model.config.use_cache = False

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # If resuming, load beacon-only weights before wrapping with PEFT.
        if args.resume_from_checkpoint:
            _load_beacon_weights_if_present(model, args.resume_from_checkpoint)

        # Freeze base params except beacons (+ optional lm_head), then add LoRA adapters.
        freeze_non_beacon_parameters(model, train_lm_head=args.train_lm_head)

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        stats = _count_trainable_params(model)
        logger.info("Total params: %.2fM | Trainable params: %.2fM", stats["total_m"], stats["trainable_m"])

        data_collator = BeaconDataCollator(tokenizer=tokenizer)

        n_gpus = torch.cuda.device_count()
        is_distributed = n_gpus > 1

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
            "dataloader_num_workers": n_gpus if is_distributed else 0,
            "ddp_find_unused_parameters": False,
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
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules,
                "bias": args.lora_bias,
            },
        }
        metrics_callback = JsonMetricsCallback(metrics_json_path, params_for_json)

        # Keep a reference to the underlying base model (inside PEFT) for beacon-weight export.
        base_model_ref = model.get_base_model() if hasattr(model, "get_base_model") else model
        beacon_weights_callback = BeaconWeightsCallback(base_model_ref)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
            callbacks=[metrics_callback, beacon_weights_callback],
        )

        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.output_dir)

    finally:
        if log_file is not None:
            log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()

