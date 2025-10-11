import argparse
import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from modeling_qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 beacon projections on multi-turn chat data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="lmsys-chat-turn-ge-3.jsonl",
        help="Path to the JSONL file that stores multi-turn conversations.",
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
    data_path: str,
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

    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info("Loaded %d raw conversations", len(dataset))

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        input_ids_batch: List[List[int]] = []
        attention_masks_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for turns in batch["conversation"]:
            if not isinstance(turns, list) or len(turns) == 0:
                continue

            messages = build_messages(turns, system_prompt)
            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokenised = tokenizer(
                chat,
                max_length=max_length,
                padding=False,
                truncation=True,
            )
            input_ids = tokenised["input_ids"]
            attention_mask = tokenised["attention_mask"]
            if len(input_ids) < 2:
                continue

            input_ids_batch.append(input_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(input_ids.copy())

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


def main() -> None:
    args = parse_args()
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
        data_path=args.data_path,
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(args.output_dir, exist_ok=True)

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
        "report_to": "none",
        "dataloader_pin_memory": True,
        "dataloader_drop_last": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "optim": "adamw_torch",
        "seed": args.seed,
    }
    if eval_dataset is not None:
        training_kwargs["eval_steps"] = args.save_steps

    supported_kwargs = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    unsupported = [key for key in list(training_kwargs.keys()) if key not in supported_kwargs]
    for key in unsupported:
        logger.warning("Dropping unsupported TrainingArguments argument: %s", key)
        training_kwargs.pop(key)

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete. Checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
