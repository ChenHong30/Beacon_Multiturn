#!/usr/bin/env python3
"""
Beacon Multi-turn Training with Distillation Loss

核心改进：添加Distillation Loss来增强beacon的训练信号

原理：
1. 对同一输入，分别计算：
   - Teacher output: 不插入beacon，使用原始完整attention
   - Student output: 插入beacon，使用beacon mask（历史body被遮蔽）
2. 让student的输出逼近teacher的输出
3. 这样beacon被迫学会"保存"历史信息，否则student无法产生正确输出

Loss = alpha * CE_loss + (1-alpha) * Distillation_loss

其中：
- CE_loss: 标准的CrossEntropy loss（只在assistant token上）
- Distillation_loss: KL散度 或 MSE（在所有位置上）
- alpha: 权重系数，建议0.5-0.7
"""

import os
import sys
import json
import math
import shlex
import logging
import inspect
import argparse
from typing import Any, Dict, List, Optional, TextIO, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

# 导入自定义模型
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args():
    parser = argparse.ArgumentParser(description="Beacon training with distillation loss")

    # 基本参数
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processed-cache-dir", type=str, default=None)

    # 训练参数
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--system-prompt", type=str, default="")

    # 精度
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Beacon参数
    parser.add_argument("--num-beacons", type=int, default=64)
    parser.add_argument("--train-beacon-only", action="store_true")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Distillation参数（新增）
    parser.add_argument("--distill-alpha", type=float, default=0.5,
                        help="Weight for CE loss. Distillation loss weight = 1 - alpha")
    parser.add_argument("--distill-temperature", type=float, default=2.0,
                        help="Temperature for distillation softmax")
    parser.add_argument("--distill-loss-type", type=str, default="kl",
                        choices=["kl", "mse", "cosine"],
                        help="Type of distillation loss")

    return parser.parse_args()


class DistillationTrainer(Trainer):
    """
    自定义Trainer，添加Distillation Loss
    """

    def __init__(self, *args, distill_alpha=0.5, distill_temperature=2.0,
                 distill_loss_type="kl", **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.distill_loss_type = distill_loss_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算混合Loss = alpha * CE_loss + (1-alpha) * Distill_loss
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        # ========== Step 1: Teacher Forward (无beacon) ==========
        # 禁用beacon，获取teacher的logits
        with torch.no_grad():
            model.model._original_seq_length = None
            model.model._compressed_seq_length = None

            teacher_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,  # 不计算loss
                use_cache=False,
                enable_beacon_compression=False,  # 关键：禁用beacon
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits.detach()
            teacher_hidden = teacher_outputs.hidden_states[-1].detach()  # 最后一层hidden

        # ========== Step 2: Student Forward (有beacon) ==========
        model.model._original_seq_length = None
        model.model._compressed_seq_length = None

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # 会自动计算CE loss
            use_cache=False,
            enable_beacon_compression=True,  # 关键：启用beacon
            output_hidden_states=True,
        )
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states[-1]

        # ========== Step 3: 计算CE Loss ==========
        # student_outputs.loss 已经是CE loss（只在labels!=-100的位置）
        ce_loss = student_outputs.loss

        # ========== Step 4: 计算Distillation Loss ==========
        # 注意：teacher和student的序列长度可能不同（因为beacon插入）
        # 我们只在原始token位置计算distillation loss

        # 获取原始序列长度（teacher的长度）
        teacher_seq_len = teacher_logits.shape[1]

        # Student的logits需要对齐到teacher
        # 这需要知道哪些位置是原始token，哪些是beacon
        # 简化方案：只对最后N个token（当前轮次）计算distillation
        # 因为当前轮次在两种模式下位置是对应的

        # 更简单的方案：对teacher和student的最后token的hidden state做distillation
        # 因为最后token的输出直接影响生成质量

        if self.distill_loss_type == "kl":
            # KL散度 on logits (只用最后一个token)
            teacher_probs = F.softmax(teacher_logits[:, -1, :] / self.distill_temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits[:, -1, :] / self.distill_temperature, dim=-1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            distill_loss = distill_loss * (self.distill_temperature ** 2)  # Scale by T^2

        elif self.distill_loss_type == "mse":
            # MSE on hidden states (最后一个token)
            distill_loss = F.mse_loss(student_hidden[:, -1, :], teacher_hidden[:, -1, :])

        elif self.distill_loss_type == "cosine":
            # Cosine similarity loss (最后一个token)
            cos_sim = F.cosine_similarity(student_hidden[:, -1, :], teacher_hidden[:, -1, :], dim=-1)
            distill_loss = (1 - cos_sim).mean()  # 1 - cos_sim 作为loss

        # ========== Step 5: 混合Loss ==========
        total_loss = self.distill_alpha * ce_loss + (1 - self.distill_alpha) * distill_loss

        # 记录各部分loss（用于监控）
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"Step {self.state.global_step}: CE={ce_loss.item():.4f}, "
                       f"Distill={distill_loss.item():.4f}, Total={total_loss.item():.4f}")

        if return_outputs:
            return total_loss, student_outputs
        return total_loss


def freeze_non_beacon_parameters(model, train_lm_head: bool = False):
    """冻结非beacon参数"""
    beacon_param_keywords = [
        "beacon_embedding",
        "beacon_position_embedding",
        "beacon_q_proj",
        "beacon_k_proj",
        "beacon_v_proj",
        "beacon_o_proj",
    ]

    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        is_beacon = any(kw in name for kw in beacon_param_keywords)
        is_lm_head = "lm_head" in name

        if is_beacon or (train_lm_head and is_lm_head):
            param.requires_grad = True
            trainable_params += param.numel()
            logger.info(f"Training: {name} ({param.numel():,} params)")
        else:
            param.requires_grad = False

    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} "
                f"({100*trainable_params/total_params:.2f}%)")


def normalise_role(role: str) -> str:
    role = role.lower()
    if role not in ALLOWED_ROLES:
        if role in {"assistant-prefill", "gpt"}:
            return "assistant"
        return "user"
    return role


def build_messages(turns, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in turns:
        role = normalise_role(turn.get("role"))
        content = turn.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        messages.append({"role": role, "content": content})
    return messages


def prepare_dataset(data_paths, tokenizer, max_length, system_prompt,
                    eval_ratio, seed, processed_cache_dir):
    """准备数据集（与原train_beacon.py相同）"""
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
    all_conversations = []

    for data_path in data_paths:
        logger.info("Loading dataset from %s", data_path)
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info("Loaded %d raw conversations from %s", len(dataset), data_path)

        for example in dataset:
            standard_conv = []
            for turn in example["conversation"]:
                standard_turn = {
                    "role": turn.get("role", ""),
                    "content": turn.get("content", "")
                }
                standard_conv.append(standard_turn)
            all_conversations.append({"conversation": standard_conv})

    from datasets import Dataset as HFDataset
    dataset = HFDataset.from_list(all_conversations)
    logger.info("Combined dataset has %d raw conversations", len(dataset))

    newline_token_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    newline_token_id = newline_token_ids[-1] if newline_token_ids else None
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def preprocess(batch):
        input_ids_batch = []
        attention_masks_batch = []
        labels_batch = []

        for turns in batch["conversation"]:
            if not isinstance(turns, list) or len(turns) == 0:
                continue

            messages = build_messages(turns, system_prompt)
            expanded_ids = []
            expanded_labels = []

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
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenising",
    )

    processed = processed.filter(lambda x: len(x["input_ids"]) > 0)
    logger.info("Final dataset has %d examples", len(processed))

    if processed_cache_dir:
        os.makedirs(processed_cache_dir, exist_ok=True)
        processed.save_to_disk(processed_cache_dir)
        logger.info("Saved processed dataset to %s", processed_cache_dir)

    if eval_ratio > 0:
        split = processed.train_test_split(test_size=eval_ratio, seed=seed)
        return split["train"], split["test"]

    return processed, None


class BeaconDataCollator:
    """数据整理器"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(self, features):
        max_length = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

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

            padded_input_ids = input_ids + [self.pad_token_id] * pad_len
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


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    logger.info("="*60)
    logger.info("Beacon Training with Distillation Loss")
    logger.info("="*60)
    logger.info(f"Distillation alpha: {args.distill_alpha}")
    logger.info(f"Distillation temperature: {args.distill_temperature}")
    logger.info(f"Distillation loss type: {args.distill_loss_type}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = prepare_dataset(
        data_paths=args.data_paths,
        tokenizer=tokenizer,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        processed_cache_dir=args.processed_cache_dir,
    )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.num_beacons_per_segment = args.num_beacons
    logger.info(f"Using {args.num_beacons} beacon tokens per segment")

    if config.model_type == "qwen2":
        model_cls = Qwen2ForCausalLM
    elif config.model_type == "qwen3":
        model_cls = Qwen3ForCausalLM
    else:
        logger.warning(f"Unknown model type: {config.model_type}. Defaulting to Qwen3.")
        model_cls = Qwen3ForCausalLM

    model = model_cls.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map=None,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.train_beacon_only:
        freeze_non_beacon_parameters(model, train_lm_head=args.train_lm_head)

    data_collator = BeaconDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_strategy="steps" if eval_dataset else "no",
        logging_dir=logging_dir,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # 使用自定义的DistillationTrainer
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        distill_loss_type=args.distill_loss_type,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
