#!/bin/bash
# Beacon Training with Distillation Loss
#
# 核心改进：添加Distillation Loss增强beacon训练信号
#
# Usage:
#   ./train_beacon_distill.sh
#
#   # 自定义distillation参数
#   DISTILL_ALPHA=0.6 DISTILL_TEMP=2.0 ./train_beacon_distill.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 基本参数
MODEL_PATH="${MODEL_PATH:-/home/hkustgz/model_weights/Qwen3-0.6B}"
DATA_PATHS="${DATA_PATHS:-ultrachat-200k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/beacon-distill-v1}"
PROCESSED_CACHE_DIR="${PROCESSED_CACHE_DIR:-./runs/dataset_cache}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2}"  # 减小batch size因为需要两次forward
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"  # 提高学习率
NUM_EPOCHS="${NUM_EPOCHS:-3}"  # 增加epoch
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_BEACONS="${NUM_BEACONS:-64}"

# Distillation参数
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"      # CE loss权重 (0.5 = 各占一半)
DISTILL_TEMP="${DISTILL_TEMP:-2.0}"        # 温度 (越高越soft)
DISTILL_TYPE="${DISTILL_TYPE:-kl}"         # kl / mse / cosine

# 检测GPU数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(nvidia-smi -L | wc -l)
else
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Beacon Training with Distillation Loss"
echo "========================================"
echo "Number of GPUs: $N_GPUS"
echo "Model path: $MODEL_PATH"
echo "Data paths: $DATA_PATHS"
echo "Output dir: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Num beacons: $NUM_BEACONS"
echo ""
echo "Distillation Settings:"
echo "  Alpha (CE weight): $DISTILL_ALPHA"
echo "  Temperature: $DISTILL_TEMP"
echo "  Loss type: $DISTILL_TYPE"
echo "========================================"

if [ "$N_GPUS" -gt 1 ]; then
    echo "Starting distributed training with $N_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_port=29500 \
        train_beacon_distill.py \
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
        --distill-alpha $DISTILL_ALPHA \
        --distill-temperature $DISTILL_TEMP \
        --distill-loss-type $DISTILL_TYPE \
        --bf16 \
        --train-beacon-only \
        --gradient-checkpointing \
        "$@"
else
    echo "Starting single GPU training..."
    python train_beacon_distill.py \
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
        --distill-alpha $DISTILL_ALPHA \
        --distill-temperature $DISTILL_TEMP \
        --distill-loss-type $DISTILL_TYPE \
        --bf16 \
        --train-beacon-only \
        --gradient-checkpointing \
        "$@"
fi

echo "Training completed!"
