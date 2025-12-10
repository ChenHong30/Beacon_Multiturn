#!/bin/bash
# Beacon Multi-turn Training Script with Multi-GPU Support
#
# Usage:
#   单卡训练:
#     ./run_train.sh
#
#   多卡训练 (自动使用所有可见GPU):
#     CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_train.sh
#
#   指定GPU训练:
#     CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数 (使用你的配置)
MODEL_PATH="${MODEL_PATH:-/home/hkustgz/model_weights/Qwen3-0.6B}"
DATA_PATHS="${DATA_PATHS:-ultrachat-200k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/beacon-ft-v2-qwen3}"
PROCESSED_CACHE_DIR="${PROCESSED_CACHE_DIR:-./runs/dataset_cache}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_BEACONS="${NUM_BEACONS:-16}"
NUM_SINKS="${NUM_SINKS:-4}"

# 检测GPU数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(nvidia-smi -L | wc -l)
else
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Beacon Multi-turn Training"
echo "========================================"
echo "Number of GPUs: $N_GPUS"
echo "Model path: $MODEL_PATH"
echo "Data paths: $DATA_PATHS"
echo "Output dir: $OUTPUT_DIR"
echo "Cache dir: $PROCESSED_CACHE_DIR"
echo "Max length: $MAX_LENGTH"
echo "Batch size per device: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "Warmup ratio: $WARMUP_RATIO"
echo "Epochs: $NUM_EPOCHS"
echo "Num beacons: $NUM_BEACONS"
echo "Num sinks: $NUM_SINKS"
echo "========================================"

if [ "$N_GPUS" -gt 1 ]; then
    echo "Starting distributed training with $N_GPUS GPUs..."
    # 使用 torchrun 进行分布式训练 (推荐，比 python -m torch.distributed.launch 更新)
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_port=29500 \
        train_beacon.py \
        --model-path "$MODEL_PATH" \
        --data-paths $DATA_PATHS \
        --output-dir "$OUTPUT_DIR" \
        --processed-cache-dir "$PROCESSED_CACHE_DIR" \
        --max-length $MAX_LENGTH \
        --per-device-train-batch-size $BATCH_SIZE \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --learning-rate $LEARNING_RATE \
        --warmup-ratio $WARMUP_RATIO \
        --num-epochs $NUM_EPOCHS \
        --save-steps $SAVE_STEPS \
        --num-beacons $NUM_BEACONS \
        --num-sinks $NUM_SINKS \
        --bf16 \
        --train-beacon-only \
        --train-lm-head \
        --gradient-checkpointing \
        "$@"
else
    echo "Starting single GPU training..."
    python train_beacon.py \
        --model-path "$MODEL_PATH" \
        --data-paths $DATA_PATHS \
        --output-dir "$OUTPUT_DIR" \
        --processed-cache-dir "$PROCESSED_CACHE_DIR" \
        --max-length $MAX_LENGTH \
        --per-device-train-batch-size $BATCH_SIZE \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --learning-rate $LEARNING_RATE \
        --warmup-ratio $WARMUP_RATIO \
        --num-epochs $NUM_EPOCHS \
        --save-steps $SAVE_STEPS \
        --num-beacons $NUM_BEACONS \
        --num-sinks $NUM_SINKS \
        --bf16 \
        --train-beacon-only \
        --train-lm-head \
        --gradient-checkpointing \
        "$@"
fi

echo "Training completed!"
