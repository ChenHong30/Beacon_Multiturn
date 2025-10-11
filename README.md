# Beacon Multiturn Fine-tuning

## 数据与权重
- 多轮对话数据保存在 `lmsys-chat-turn-ge-3.jsonl`
- 预训练权重目录：`Qwen2-0.5B-Instruct/`

## 训练脚本
`train_beacon.py` 会：
- 用 tokenizer 的 `apply_chat_template` 将每条对话打包为完整的多轮上下文；
- 让模型内部的 `parse_multiturn_dialogue` 在预填充阶段插入 beacon token，确保 Q/K/V 投影被更新；
- 默认执行语言模型交叉熵训练。

### 启动参数
常用参数说明：
- `--data-path`：JSONL 数据路径，默认 `lmsys-chat-turn-ge-3.jsonl`
- `--model-path`：初始化模型的 checkpoint 目录，默认 `./Qwen2-0.5B-Instruct`
- `--output-dir`：保存 finetune 结果的目录，默认 `./beacon-finetune`
- `--processed-cache-dir`：预处理后的数据集缓存目录（可选，配置后首次运行会写入，下次直接加载）
- `--max-length`：单条样本的最大 token 数（截断），默认 2048
- `--num-epochs`：训练轮数
- `--learning-rate`、`--weight-decay`、`--warmup-ratio`：优化器相关超参
- `--per-device-train-batch-size`、`--gradient-accumulation-steps`：有效 batch 控制
- `--bf16` / `--fp16`：混合精度开关（二选一），`--gradient-checkpointing`：梯度检查点
- `--train-beacon-only`：只训练 `beacon_q/k/v_proj` 与 beacon embedding；配合 `--train-lm-head` 可额外训练输出头
- `--eval-ratio`：切分验证集比例（默认 2%）；设为 0 则不做评估

完整参数可通过 `python train_beacon.py --help` 查看。

### 启动命令示例
```bash
python train_beacon.py \
  --data-path lmsys-chat-turn-ge-3.jsonl \
  --model-path ./Qwen2-0.5B-Instruct \
  --output-dir ./runs/beacon-ft \
  --processed-cache-dir ./runs/dataset-cache \
  --num-epochs 1 \
  --learning-rate 5e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --bf16 \
  --train-beacon-only \
  --train-lm-head
```

不建议启用 flash attention（部分环境下存在兼容问题）；脚本不会强制改动 `_attn_implementation`。

训练完成后，新的权重会保存在 `--output-dir`，使用 `Qwen2ForCausalLM.from_pretrained(<output_dir>)` 加载即可；beacon 投影矩阵会直接从 checkpoint 中恢复，不再随机初始化。***
