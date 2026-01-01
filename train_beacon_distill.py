import argparse
import inspect
import json
import logging
import os
import shlex
import sys
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

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
    parser = argparse.ArgumentParser(description="Distill beacon compression using a teacher model.")
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
        default="./Qwen3-0.6B",
        help="Checkpoint directory used to initialise the student model.",
    )
    parser.add_argument(
        "--teacher-model-path",
        type=str,
        default=None,
        help="Checkpoint directory for the teacher model (defaults to --model-path when unset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./beacon-distill",
        help="Directory where the distilled checkpoint will be written.",
    )
    parser.add_argument(
        "--processed-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache the processed tokenised dataset.",
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
        default=4096,
        help="Maximum number of tokens per packed sample after tokenisation.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.02,
        help="Portion of examples reserved for evaluation (0 disables evaluation).",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay applied to trainable params.")
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
        "--beacon-recon-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary beacon reconstruction loss (default: 0.0, disabled).",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=1.0,
        help="Temperature for distillation soft targets.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="Weight of the distillation KL loss.",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.0,
        help="Weight of the student cross-entropy loss.",
    )
    parser.add_argument(
        "--beacon-attn-weight",
        type=float,
        default=0.0,
        help="Weight for beacon attention regularization loss (encourages attending to beacons).",
    )
    parser.add_argument(
        "--hidden-distill-weight",
        type=float,
        default=0.0,
        help="Weight for hidden states distillation loss.",
    )
    parser.add_argument(
        "--hidden-distill-layer",
        type=int,
        default=-1,
        help="Which layer's hidden states to distill (-1 for last layer).",
    )
    parser.add_argument(
        "--min-beacon-attn",
        type=float,
        default=0.1,
        help="Minimum attention ratio to beacons (for beacon attention regularization).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder to resume training from.",
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
            standard_conv = []
            for turn in example["conversation"]:
                standard_turn = {
                    "role": turn.get("role", ""),
                    "content": turn.get("content", ""),
                }
                standard_conv.append(standard_turn)
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


class DistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model: torch.nn.Module,
        distill_weight: float,
        ce_weight: float,
        temperature: float,
        beacon_attn_weight: float = 0.0,
        hidden_distill_weight: float = 0.0,
        hidden_distill_layer: int = -1,
        min_beacon_attn: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        self.ce_weight = ce_weight
        self.temperature = temperature
        self.beacon_attn_weight = beacon_attn_weight
        self.hidden_distill_weight = hidden_distill_weight
        self.hidden_distill_layer = hidden_distill_layer
        self.min_beacon_attn = min_beacon_attn
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    def _build_distill_mask(
        self,
        student_model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        student_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, modified_input_ids, _, _ = student_model.model.parse_multiturn_dialogue(input_ids, labels)

        if modified_input_ids.shape[1] != student_seq_len:
            modified_input_ids = modified_input_ids[:, :student_seq_len]

        beacon_token_id = student_model.model.beacon_token_id
        batch_size = modified_input_ids.shape[0]
        mapping = torch.full(
            (batch_size, student_seq_len),
            -1,
            dtype=torch.long,
            device=modified_input_ids.device,
        )
        distill_mask = torch.zeros(
            (batch_size, student_seq_len),
            dtype=torch.bool,
            device=modified_input_ids.device,
        )

        orig_len = input_ids.shape[1]
        for b in range(batch_size):
            mod_ids = modified_input_ids[b].tolist()
            orig_labels = labels[b].tolist() if labels is not None else None
            orig_attn = attention_mask[b].tolist() if attention_mask is not None else None
            orig_idx = 0
            for j, token_id in enumerate(mod_ids):
                if token_id == beacon_token_id:
                    continue
                if orig_idx >= orig_len:
                    break
                mapping[b, j] = orig_idx
                if orig_labels is not None:
                    if orig_labels[orig_idx] != -100:
                        distill_mask[b, j] = True
                elif orig_attn is not None:
                    if orig_attn[orig_idx] == 1:
                        distill_mask[b, j] = True
                else:
                    distill_mask[b, j] = True
                orig_idx += 1

        return distill_mask, mapping

    def _compute_kd_loss(
        self,
        student_model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_seq_len = student_logits.shape[1]
        distill_mask, mapping = self._build_distill_mask(
            student_model=student_model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            student_seq_len=student_seq_len,
        )

        if distill_mask.sum().item() == 0:
            return student_logits.new_tensor(0.0)

        vocab_size = student_logits.shape[-1]
        student_flat = student_logits.view(-1, vocab_size)
        flat_mask = distill_mask.view(-1)
        mapping_flat = mapping.view(-1)[flat_mask]

        valid = mapping_flat >= 0
        if not valid.any():
            return student_logits.new_tensor(0.0)

        student_sel = student_flat[flat_mask][valid]
        batch_size = teacher_logits.shape[0]
        batch_ids = torch.arange(batch_size, device=teacher_logits.device).unsqueeze(1)
        batch_ids = batch_ids.expand(batch_size, student_seq_len).reshape(-1)
        teacher_sel = teacher_logits[batch_ids[flat_mask][valid], mapping_flat[valid]]

        temperature = max(self.temperature, 1e-5)
        student_log_probs = F.log_softmax(student_sel.float() / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_sel.float() / temperature, dim=-1)
        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (temperature * temperature)
        return kd_loss

    def _compute_beacon_attn_loss(
        self,
        student_model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        attentions: tuple,
    ) -> torch.Tensor:
        """
        计算beacon注意力正则化损失。
        鼓励当前轮次的token更多地attend到历史beacons。
        """
        if attentions is None or len(attentions) == 0:
            return input_ids.new_tensor(0.0, dtype=torch.float)

        # 解析对话结构，获取beacon位置和当前轮次位置
        with torch.no_grad():
            segment_bounds, modified_input_ids, _, _ = student_model.model.parse_multiturn_dialogue(
                input_ids, labels
            )

        beacon_token_id = student_model.model.beacon_token_id
        batch_size, seq_len = modified_input_ids.shape

        # 找到beacon位置和当前轮次起始位置
        beacon_mask = (modified_input_ids == beacon_token_id)  # [B, S]
        
        total_loss = input_ids.new_tensor(0.0, dtype=torch.float)
        valid_count = 0

        for b in range(batch_size):
            bounds = segment_bounds[b]
            if len(bounds) < 2:
                continue

            # 当前轮次是最后一个segment
            current_start = bounds[-1][0]
            current_end = bounds[-1][1]
            
            # 历史beacon位置（不包括当前轮次的beacons，如果有的话）
            history_beacon_indices = []
            for seg_idx, (start, end) in enumerate(bounds[:-1]):
                seg_beacon_mask = beacon_mask[b, start:end]
                seg_beacon_positions = torch.where(seg_beacon_mask)[0] + start
                history_beacon_indices.extend(seg_beacon_positions.tolist())

            if len(history_beacon_indices) == 0:
                continue

            history_beacon_indices = torch.tensor(history_beacon_indices, device=input_ids.device)

            # 对每一层的attention计算beacon注意力
            for layer_idx, layer_attn in enumerate(attentions):
                # layer_attn: [B, num_heads, S, S]
                if layer_attn.shape[2] != seq_len or layer_attn.shape[3] != seq_len:
                    continue

                # 当前轮次tokens对历史beacons的注意力
                # 取当前轮次的query positions
                current_attn = layer_attn[b, :, current_start:current_end, :]  # [H, current_len, S]
                
                # 对历史beacons的注意力
                beacon_attn = current_attn[:, :, history_beacon_indices]  # [H, current_len, num_beacons]
                avg_beacon_attn = beacon_attn.mean()  # 平均注意力

                # 如果注意力低于阈值，产生loss
                if avg_beacon_attn < self.min_beacon_attn:
                    # Hinge loss: max(0, min_attn - actual_attn)
                    layer_loss = self.min_beacon_attn - avg_beacon_attn
                    total_loss = total_loss + layer_loss
                    valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss

    def _compute_hidden_distill_loss(
        self,
        student_model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        student_hidden_states: tuple,
        teacher_hidden_states: tuple,
    ) -> torch.Tensor:
        """
        计算hidden states蒸馏损失。
        在当前轮次位置，学生和教师的hidden states应该相似。
        使用cosine similarity loss，对scale不敏感。
        """
        if student_hidden_states is None or teacher_hidden_states is None:
            return input_ids.new_tensor(0.0, dtype=torch.float)

        # 选择要蒸馏的层
        layer_idx = self.hidden_distill_layer
        if layer_idx < 0:
            layer_idx = len(student_hidden_states) + layer_idx
        
        if layer_idx < 0 or layer_idx >= len(student_hidden_states):
            return input_ids.new_tensor(0.0, dtype=torch.float)

        student_hidden = student_hidden_states[layer_idx]  # [B, student_seq_len, D]
        teacher_hidden = teacher_hidden_states[layer_idx]  # [B, teacher_seq_len, D]

        # 解析对话结构，建立学生到教师的位置映射
        with torch.no_grad():
            segment_bounds, modified_input_ids, _, _ = student_model.model.parse_multiturn_dialogue(
                input_ids, labels
            )

        beacon_token_id = student_model.model.beacon_token_id
        batch_size = input_ids.shape[0]
        teacher_seq_len = teacher_hidden.shape[1]
        student_seq_len = student_hidden.shape[1]

        total_loss = input_ids.new_tensor(0.0, dtype=torch.float)
        valid_count = 0

        for b in range(batch_size):
            # 建立学生位置到教师位置的映射（跳过beacon tokens）
            mod_ids = modified_input_ids[b].tolist()
            teacher_idx = 0
            
            student_positions = []
            teacher_positions = []
            
            for s_idx, token_id in enumerate(mod_ids):
                if token_id == beacon_token_id:
                    continue
                if teacher_idx >= teacher_seq_len:
                    break
                if s_idx >= student_seq_len:
                    break
                    
                # 只对有标签的位置（assistant回复）进行蒸馏
                if labels is not None and labels[b, teacher_idx].item() != -100:
                    student_positions.append(s_idx)
                    teacher_positions.append(teacher_idx)
                
                teacher_idx += 1

            if len(student_positions) == 0:
                continue

            student_positions = torch.tensor(student_positions, device=input_ids.device)
            teacher_positions = torch.tensor(teacher_positions, device=input_ids.device)

            # 提取对应位置的hidden states
            student_h = student_hidden[b, student_positions, :]  # [N, D]
            teacher_h = teacher_hidden[b, teacher_positions, :]  # [N, D]

            # Cosine similarity loss: 1 - cos_sim (越相似loss越小)
            cos_sim = F.cosine_similarity(student_h.float(), teacher_h.float(), dim=-1)  # [N]
            loss = (1 - cos_sim).mean()
            total_loss = total_loss + loss
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        # 判断是否需要额外输出
        need_attentions = self.beacon_attn_weight > 0.0
        need_hidden_states = self.hidden_distill_weight > 0.0

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                enable_beacon_compression=False,
                output_hidden_states=need_hidden_states,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states if need_hidden_states else None

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            enable_beacon_compression=True,
            output_attentions=need_attentions,
            output_hidden_states=need_hidden_states,
        )
        student_logits = student_outputs.logits
        ce_loss = student_outputs.loss if labels is not None else None
        student_attentions = student_outputs.attentions if need_attentions else None
        student_hidden_states = student_outputs.hidden_states if need_hidden_states else None

        student_model = self._unwrap_model(model)
        kd_loss = self._compute_kd_loss(
            student_model=student_model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

        loss = student_logits.new_tensor(0.0)
        if self.distill_weight > 0.0:
            loss = loss + self.distill_weight * kd_loss
        if self.ce_weight > 0.0 and ce_loss is not None:
            loss = loss + self.ce_weight * ce_loss

        # 方案1: Beacon注意力正则化
        if self.beacon_attn_weight > 0.0 and student_attentions is not None:
            beacon_attn_loss = self._compute_beacon_attn_loss(
                student_model=student_model,
                input_ids=input_ids,
                labels=labels,
                attentions=student_attentions,
            )
            loss = loss + self.beacon_attn_weight * beacon_attn_loss

        # 方案3: Hidden states蒸馏
        if self.hidden_distill_weight > 0.0 and student_hidden_states is not None and teacher_hidden_states is not None:
            hidden_distill_loss = self._compute_hidden_distill_loss(
                student_model=student_model,
                input_ids=input_ids,
                labels=labels,
                student_hidden_states=student_hidden_states,
                teacher_hidden_states=teacher_hidden_states,
            )
            loss = loss + self.hidden_distill_weight * hidden_distill_loss

        if return_outputs:
            student_outputs.loss = loss
            return loss, student_outputs
        return loss


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

        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.num_beacons_per_segment = args.num_beacons
        config.num_sinks = args.num_sinks
        config.beacon_recon_weight = float(args.beacon_recon_weight)

        model_cls = Qwen3ForCausalLM
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
        model = model_cls.from_pretrained(
            args.model_path,
            config=config,
            tokenizer=tokenizer,
            dtype=dtype,
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

        teacher_path = args.teacher_model_path or args.model_path
        teacher_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
        teacher_model = model_cls.from_pretrained(
            teacher_path,
            config=teacher_config,
            tokenizer=tokenizer,
            dtype=dtype,
            device_map=None,
            attn_implementation="eager",
        )
        teacher_model.config.use_cache = False
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

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
        teacher_model.to(training_args.device)

        params_for_json = {
            "cli_args": sanitize_for_json(vars(args)),
            "training_arguments": sanitize_for_json(training_args.to_dict()),
        }
        metrics_callback = JsonMetricsCallback(metrics_json_path, params_for_json)

        trainer = DistillTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
            callbacks=[metrics_callback],
            teacher_model=teacher_model,
            distill_weight=args.distill_weight,
            ce_weight=args.ce_weight,
            temperature=args.distill_temperature,
            beacon_attn_weight=args.beacon_attn_weight,
            hidden_distill_weight=args.hidden_distill_weight,
            hidden_distill_layer=args.hidden_distill_layer,
            min_beacon_attn=args.min_beacon_attn,
        )

        if args.resume_from_checkpoint is not None:
            logger.info("Resuming training from checkpoint: %s", args.resume_from_checkpoint)
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
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
