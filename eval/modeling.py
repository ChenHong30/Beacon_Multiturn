from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer, pipeline

from modeling_qwen3 import Qwen3ForCausalLM


def create_generation_pipeline(
    model_path: str,
    device_id: int = 0,
    *,
    trust_remote_code: bool = True,
    fix_mistral_regex: Optional[bool] = None,
    attn_implementation: str = "eager",
    log: bool = True,
):
    if log:
        print(f"Creating generation pipeline for: {model_path}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    pipe_device = device_id if torch.cuda.is_available() else -1
    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    if fix_mistral_regex is not None:
        tokenizer_kwargs["fix_mistral_regex"] = fix_mistral_regex

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=pipe_device,
        model_kwargs={
            "attn_implementation": attn_implementation,
        },
    )

    if log:
        print(f"Pipeline created. Model loaded on {pipe.device} (tokenizer available)")

    return pipe, tokenizer, pipe.device


def generate_text_with_pipeline(
    pipe,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float],
    return_full_text: bool = False,
    clean_up_tokenization_spaces: bool = True,
    use_cache: bool = True,
) -> str:
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
        "return_full_text": return_full_text,
        "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    output = pipe(prompt_text, **gen_kwargs)
    return output[0]["generated_text"].strip()


def load_beacon_model(
    model_path: str,
    device_id: int = 0,
    *,
    num_sinks: int = 0,
    tokenizer=None,
    attn_implementation: str = "eager",
    strict_num_beacons: bool = True,
):
    print(f"Loading Beacon model: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    config.num_sinks = num_sinks

    if strict_num_beacons:
        num_beacons = config.num_beacons_per_segment
    else:
        num_beacons = getattr(config, "num_beacons_per_segment", None)

    print(f"num_sinks = {config.num_sinks}")
    print(f"num_beacons_per_segment = {num_beacons}")

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        "config": config,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": None,
        "attn_implementation": attn_implementation,
    }
    if tokenizer is not None:
        model_kwargs["tokenizer"] = tokenizer

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    model = model.to(device)
    model.eval()

    print(f"Beacon model loaded on {device}")
    return model, device


def encode_prompt(tokenizer, prompt_text: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def generate_beacon_response(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float] = None,
    apply_temperature_if_sampled: bool = False,
    enable_beacon_compression: bool = True,
) -> str:
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "enable_beacon_compression": enable_beacon_compression,
    }

    if (not apply_temperature_if_sampled) or do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    input_len = input_ids.shape[1]
    new_tokens = output_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text
