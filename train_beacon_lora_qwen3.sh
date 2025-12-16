#!/bin/bash

# Example launcher for LoRA + Beacon training on Qwen3.
# - Trains LoRA adapters on the base model weights
# - Also finetunes all beacon parameters (proj/norm/embedding)
# - Beacon compression stays enabled via the custom model forward path

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parameters (edit these)
MODEL_PATH="/data/hkustgz/model_weight/16_beacon_4_sink"   # beacon-trained checkpoint or base model
DATA_PATHS="lmsys-chat-sampled-60k.enqa.tokle4096.jsonl"                    # one or more JSONL paths (space separated)
OUTPUT_DIR="/data/hkustgz/model_weight/16_beacon_4_sink_lora"
PROCESSED_CACHE_DIR="./runs/dataset_cache_lora"

MAX_LENGTH="4096"
BATCH_SIZE="1"
GRAD_ACCUM="2"
LEARNING_RATE="2e-4"
NUM_EPOCHS="1"
WARMUP_RATIO="0.03"
SAVE_STEPS="500"

NUM_BEACONS="16"
NUM_SINKS="4"

# LoRA params
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_BIAS="none"

# Determine number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  N_GPUS=$(nvidia-smi -L | wc -l)
else
  N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "LoRA + Beacon Training (Qwen3)"
echo "========================================"
echo "GPUs                : $N_GPUS"
echo "MODEL_PATH          : $MODEL_PATH"
echo "DATA_PATHS          : $DATA_PATHS"
echo "OUTPUT_DIR          : $OUTPUT_DIR"
echo "CACHE_DIR           : $PROCESSED_CACHE_DIR"
echo "MAX_LENGTH          : $MAX_LENGTH"
echo "BATCH_SIZE/dev      : $BATCH_SIZE"
echo "GRAD_ACCUM          : $GRAD_ACCUM"
echo "LEARNING_RATE       : $LEARNING_RATE"
echo "WARMUP_RATIO        : $WARMUP_RATIO"
echo "EPOCHS              : $NUM_EPOCHS"
echo "NUM_BEACONS         : $NUM_BEACONS"
echo "NUM_SINKS           : $NUM_SINKS"
echo "LORA_R              : $LORA_R"
echo "LORA_ALPHA          : $LORA_ALPHA"
echo "LORA_DROPOUT        : $LORA_DROPOUT"
echo "LORA_TARGET_MODULES : $LORA_TARGET_MODULES"
echo "LORA_BIAS           : $LORA_BIAS"
echo "========================================"

if [ "$N_GPUS" -gt 1 ]; then
  torchrun \
    --nproc_per_node="$N_GPUS" \
    --master_port=29501 \
    train_lora_beacon.py \
    --model-path "$MODEL_PATH" \
    --data-paths $DATA_PATHS \
    --output-dir "$OUTPUT_DIR" \
    --processed-cache-dir "$PROCESSED_CACHE_DIR" \
    --max-length "$MAX_LENGTH" \
    --per-device-train-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --num-epochs "$NUM_EPOCHS" \
    --save-steps "$SAVE_STEPS" \
    --num-beacons "$NUM_BEACONS" \
    --num-sinks "$NUM_SINKS" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --lora-bias "$LORA_BIAS" \
    --bf16 \
    --gradient-checkpointing \
    "$@"
else
  python train_lora_beacon.py \
    --model-path "$MODEL_PATH" \
    --data-paths $DATA_PATHS \
    --output-dir "$OUTPUT_DIR" \
    --processed-cache-dir "$PROCESSED_CACHE_DIR" \
    --max-length "$MAX_LENGTH" \
    --per-device-train-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --num-epochs "$NUM_EPOCHS" \
    --save-steps "$SAVE_STEPS" \
    --num-beacons "$NUM_BEACONS" \
    --num-sinks "$NUM_SINKS" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --lora-bias "$LORA_BIAS" \
    --bf16 \
    --gradient-checkpointing \
    "$@"
fi

echo "Training completed!"

