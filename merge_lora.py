#!/usr/bin/env python3
"""
Merge LoRA adapters with base model and save the merged model.
Also handles beacon weights if present.

Usage:
    python merge_lora.py --adapter-dir /path/to/adapter --model-path /path/to/base/model
    python merge_lora.py --adapter-dir /path/to/adapter  # Uses model_id from adapter config
    python merge_lora.py --adapter-dir /path/to/adapter --device cuda
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model and save merged model."
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Directory containing LoRA adapter (adapter_config.json and adapter_model.bin)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to base model. If not provided, will read from adapter config's base_model_name_or_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged model. If not provided, uses adapter_dir + '_merged'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for merging (cpu, cuda, etc.)",
    )
    return parser.parse_args()


def get_base_model_path(adapter_dir: str) -> str:
    """Extract base model path from adapter config if model_path not provided."""
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    base_model_path = config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(
            f"base_model_name_or_path not found in {adapter_config_path}. "
            "Please provide --model-path explicitly."
        )
    
    logger.info("Read base_model_name_or_path from adapter config: %s", base_model_path)
    return base_model_path


def has_beacon_weights(adapter_dir: str) -> bool:
    """Check if beacon weights exist in the adapter directory."""
    beacon_weights_path = os.path.join(adapter_dir, "beacon_weights.safetensors")
    beacon_weights_pt_path = os.path.join(adapter_dir, "beacon_weights.safetensors.pt")
    return os.path.isfile(beacon_weights_path) or os.path.isfile(beacon_weights_pt_path)


def load_beacon_weights(adapter_dir: str) -> dict:
    """Load beacon weights from adapter directory."""
    beacon_weights_path = os.path.join(adapter_dir, "beacon_weights.safetensors")
    beacon_weights_pt_path = os.path.join(adapter_dir, "beacon_weights.safetensors.pt")
    
    if os.path.isfile(beacon_weights_path):
        try:
            from safetensors.torch import load_file as safe_load_file
            beacon_state = safe_load_file(beacon_weights_path, device="cpu")
            logger.info("Loaded beacon weights from %s", beacon_weights_path)
            return beacon_state
        except Exception as e:
            logger.warning("Failed to load safetensors beacon weights: %s", e)
    
    if os.path.isfile(beacon_weights_pt_path):
        try:
            beacon_state = torch.load(beacon_weights_pt_path, map_location="cpu")
            logger.info("Loaded beacon weights from %s", beacon_weights_pt_path)
            return beacon_state
        except Exception as e:
            logger.warning("Failed to load pytorch beacon weights: %s", e)
    
    return {}


def merge_and_save(
    adapter_dir: str,
    model_path: str,
    output_dir: str,
    device: str = "cpu",
) -> None:
    """Merge LoRA adapter with base model and save."""
    
    logger.info("=" * 60)
    logger.info("LoRA Adapter Merge")
    logger.info("=" * 60)
    logger.info("Adapter directory: %s", adapter_dir)
    logger.info("Base model path:   %s", model_path)
    logger.info("Output directory:  %s", output_dir)
    logger.info("Device:            %s", device)
    
    # Verify adapter directory
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    
    # Find adapter model file (can be .bin or .safetensors)
    adapter_model_path = None
    for fname in ["adapter_model.bin", "adapter_model.safetensors"]:
        candidate = os.path.join(adapter_dir, fname)
        if os.path.isfile(candidate):
            adapter_model_path = candidate
            logger.info("Found adapter model: %s", fname)
            break
    
    if adapter_model_path is None:
        raise FileNotFoundError(
            f"adapter_model.bin or adapter_model.safetensors not found in {adapter_dir}"
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    logger.info("Loading base model from %s...", model_path)
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        from modeling_qwen3 import Qwen3ForCausalLM
        base_model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            attn_implementation="eager",
        )
    except Exception as e:
        logger.warning("Failed to load with custom Qwen3 model, trying standard method: %s", e)
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
    
    logger.info("Base model loaded successfully")
    
    # Load beacon weights if present
    beacon_state = None
    if has_beacon_weights(adapter_dir):
        logger.info("Found beacon weights, will merge them as well")
        beacon_state = load_beacon_weights(adapter_dir)
        base_model.load_state_dict(beacon_state, strict=False)
        logger.info("Beacon weights loaded and applied")
    
    # Load and merge LoRA adapter
    logger.info("Loading LoRA adapter from %s...", adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir, device_map=device)
    logger.info("LoRA adapter loaded")
    
    logger.info("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    logger.info("LoRA adapter merged successfully")
    
    # Save merged model (same format as train_beacon.py)
    logger.info("Saving merged model to %s...", output_dir)
    merged_model.save_pretrained(output_dir)
    logger.info("Merged model saved")
    
    # Copy tokenizer (same as train_beacon.py)
    logger.info("Copying tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer copied from adapter directory")
    except Exception as e:
        logger.warning("Failed to copy tokenizer from adapter, trying base model: %s", e)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            tokenizer.save_pretrained(output_dir)
            logger.info("Tokenizer copied from base model")
        except Exception as e2:
            logger.warning("Failed to copy tokenizer: %s", e2)
    
    # Save beacon weights separately if they exist (same as train_beacon.py)
    if beacon_state:
        try:
            from safetensors.torch import save_file as safe_save_file
            beacon_weights_path = os.path.join(output_dir, "beacon_weights.safetensors")
            safe_save_file(beacon_state, beacon_weights_path)
            logger.info("Beacon weights saved to %s", beacon_weights_path)
            
            meta_path = os.path.join(output_dir, "beacon_weights.meta.json")
            meta = {
                "format": "beacon_only",
                "num_keys": len(beacon_state),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save beacon weights: %s", e)
    
    logger.info("=" * 60)
    logger.info("Merge completed successfully!")
    logger.info("Merged model saved to: %s", output_dir)
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    
    # Determine base model path
    if args.model_path is None:
        model_path = get_base_model_path(args.adapter_dir)
    else:
        model_path = args.model_path
    
    # Determine output directory
    if args.output_dir is None:
        adapter_dir_name = os.path.basename(args.adapter_dir.rstrip(os.sep))
        parent_dir = os.path.dirname(args.adapter_dir.rstrip(os.sep))
        output_dir = os.path.join(parent_dir, f"{adapter_dir_name}_merged")
    else:
        output_dir = args.output_dir
    
    try:
        merge_and_save(
            adapter_dir=args.adapter_dir,
            model_path=model_path,
            output_dir=output_dir,
            device=args.device,
        )
    except Exception as e:
        logger.error("Merge failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
