import os
import gc
from typing import List, Dict

import torch
from transformers import AutoConfig, AutoTokenizer

from modeling_qwen2 import Qwen2ForCausalLM


CHECKPOINT_PATH = "/home/catlover/projects/Beacon_Multiturn/runs/beacon-ft/checkpoint-16988"

def get_gpu_memory_usage():
    """
    获取GPU显存使用情况
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def clear_gpu_cache():
    """
    清理GPU缓存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_qwen_model(model_path: str):
    """
    加载Qwen2.5-1.5B-Instruct模型，始终加载到第一张显卡（cuda:0），
    使用本地的modeling_qwen2.py配置
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    print(f"正在加载模型: {model_path}")

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("加载配置...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print("加载模型权重...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,     # 重要：不要用"auto"
    )
    model = model.to(device)   # 显式转到cuda:0

    print("模型加载完成!")
    return model, tokenizer, config, device

def calculate_kv_cache_size(past_key_values):
    """
    计算KV cache的大小（以MB为单位）
    """
    if past_key_values is None:
        return 0
    
    total_size = 0
    for layer_idx in range(len(past_key_values.key_cache)):
        key_cache = past_key_values.key_cache[layer_idx]
        value_cache = past_key_values.value_cache[layer_idx]
        
        # 计算每个tensor的字节数
        key_size = key_cache.numel() * key_cache.element_size()
        value_size = value_cache.numel() * value_cache.element_size()
        total_size += key_size + value_size
    
    return total_size / (1024 * 1024)  # 转换为MB

def render_dialogue(dialogue: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    """
    使用tokenizer的chat模板渲染多轮对话文本
    """
    return tokenizer.apply_chat_template(
        dialogue,
        add_generation_prompt=True,
        tokenize=False,
    )


def encode_dialogue(dialogue: List[Dict[str, str]], tokenizer: AutoTokenizer, device: torch.device):
    """
    使用chat模板编码多轮对话，返回input_ids和attention_mask
    """
    encoded = tokenizer.apply_chat_template(
        dialogue,
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(encoded[0], list):
        # apply_chat_template 可能返回嵌套列表（批次），此处保证为一维
        encoded = encoded[0]
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def test_multiturn_dialogue(model, tokenizer, device):
    """
    测试多轮对话的beacon压缩功能并比较KV cache大小
    """
    print("\n开始测试多轮对话beacon压缩功能...")
    
    # 构建多轮对话格式
    dialogue = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
        {"role": "assistant", "content": "你好！我是一个AI助手，很高兴为您服务。"},
        {"role": "user", "content": "你能帮我解决数学问题吗？"},
        {"role": "assistant", "content": "当然可以！我很乐意帮您解决数学问题。"},
        {"role": "user", "content": "那么1+1等于多少？"}
    ]
    
    rendered_text = render_dialogue(dialogue, tokenizer)
    print(f"多轮对话输入:\n{rendered_text}")
    
    # 编码输入
    input_ids, attention_mask = encode_dialogue(dialogue, tokenizer, device)
    
    print(f"原始input_ids长度: {input_ids.shape[1]}")
    print(f"原始input_ids: {input_ids[0].tolist()[:50]}...")  # 只显示前50个token
    
    # 测试模型的forward pass（启用beacon压缩）
    with torch.no_grad():
        print("\n=== 启用beacon压缩 ===")
        clear_gpu_cache()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            enable_beacon_compression=True
        )
        
        # 计算压缩后的KV cache大小
        compressed_kv_size = calculate_kv_cache_size(outputs.past_key_values)
        
        if outputs.past_key_values is not None:
            print(f"压缩后的KV cache层数: {len(outputs.past_key_values.key_cache)}")
            if len(outputs.past_key_values.key_cache) > 0:
                print(f"第一层KV cache shape: {outputs.past_key_values.key_cache[0].shape}")
                print(f"压缩后KV cache大小: {compressed_kv_size:.2f}MB")
        
        # 清理缓存，准备下一个测试
        del outputs
        clear_gpu_cache()
        
        print("\n=== 禁用beacon压缩（对比） ===")
        
        outputs_no_compression = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            enable_beacon_compression=False
        )
        
        # 计算未压缩的KV cache大小
        uncompressed_kv_size = calculate_kv_cache_size(outputs_no_compression.past_key_values)
        
        if outputs_no_compression.past_key_values is not None:
            print(f"未压缩的KV cache层数: {len(outputs_no_compression.past_key_values.key_cache)}")
            if len(outputs_no_compression.past_key_values.key_cache) > 0:
                print(f"第一层KV cache shape: {outputs_no_compression.past_key_values.key_cache[0].shape}")
                print(f"未压缩KV cache大小: {uncompressed_kv_size:.2f}MB")
        
        # KV cache节省对比
        if uncompressed_kv_size > 0:
            kv_saved = uncompressed_kv_size - compressed_kv_size
            kv_saved_percent = (kv_saved / uncompressed_kv_size) * 100
            print(f"\nKV cache节省: {kv_saved:.2f}MB ({kv_saved_percent:.1f}%)")
            print(f"压缩比: {uncompressed_kv_size/compressed_kv_size:.1f}x")
        
        # 清理缓存
        del outputs_no_compression
        clear_gpu_cache()
    
    # 测试生成（使用beacon压缩）
    print("\n=== 测试生成（beacon压缩） ===")
    with torch.no_grad():
        clear_gpu_cache()
        
        # 首先进行prefill阶段，启用beacon压缩
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            enable_beacon_compression=True
        )
        
        prefill_kv_size = calculate_kv_cache_size(prefill_outputs.past_key_values)
        print(f"Prefill阶段KV cache大小: {prefill_kv_size:.2f}MB")
        
        # 使用压缩后的KV cache进行生成
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=prefill_outputs.past_key_values,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        print(f"生成完成，生成了 {generated_ids.shape[1] - input_ids.shape[1]} 个新token")
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"生成的完整文本:\n{generated_text}")
    
    # 只显示新生成的部分
    new_tokens = generated_ids[0, input_ids.shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).lstrip()
    if new_text.startswith("assistant"):
        new_text = new_text[len("assistant") :].lstrip(": \n")
    print(f"\n新生成的文本: {new_text}")
    print(f"\n新生成的文本: {new_text}")

if __name__ == "__main__":
    try:
        # 加载模型
        model, tokenizer, config, device = load_qwen_model(CHECKPOINT_PATH)

        print(f"\n模型信息:")
        print(f"模型类型: {type(model).__name__}")
        print(f"词汇表大小: {config.vocab_size}")
        print(f"隐藏层大小: {config.hidden_size}")
        print(f"注意力头数: {config.num_attention_heads}")
        print(f"层数: {config.num_hidden_layers}")
        
        # 测试多轮对话beacon压缩功能
        test_multiturn_dialogue(model, tokenizer, device)

        print("\n模型加载和测试完成！多轮对话KV压缩功能已实现。")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
