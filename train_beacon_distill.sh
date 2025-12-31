#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B/"
TEACHER_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B/"
DATA_PATHS="dataset_multiturn_generated.jsonl"
OUTPUT_DIR="/data/hkustgz/model_weight/8_beacon_4_sink_distill_generated"
PROCESSED_CACHE_DIR="./runs/dataset_cache_generated"
MAX_LENGTH="4096"
BATCH_SIZE="1"
GRAD_ACCUM="2"
LEARNING_RATE="1e-4"
NUM_EPOCHS="16"
WARMUP_RATIO="0.03"
SAVE_STEPS="500"
NUM_BEACONS="8"
NUM_SINKS="4"
BEACON_RECON_WEIGHT="1.0"
DISTILL_WEIGHT="1.0"
CE_WEIGHT="0.0"
DISTILL_TEMP="1.0"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(nvidia-smi -L | wc -l)
else
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "========================================"
echo "Beacon Distillation Training"
echo "========================================"
echo "Number of GPUs: $N_GPUS"
echo "Student model path: $MODEL_PATH"
echo "Teacher model path: $TEACHER_MODEL_PATH"
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
echo "Beacon recon weight: $BEACON_RECON_WEIGHT"
echo "Distill weight: $DISTILL_WEIGHT"
echo "CE weight: $CE_WEIGHT"
echo "Distill temperature: $DISTILL_TEMP"
echo "========================================"

if [ "$N_GPUS" -gt 1 ]; then
    echo "Starting distributed training with $N_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_port=29501 \
        train_beacon_distill.py \
        --model-path "$MODEL_PATH" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
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
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --distill-weight $DISTILL_WEIGHT \
        --ce-weight $CE_WEIGHT \
        --distill-temperature $DISTILL_TEMP \
        --bf16 \
        --train-beacon-only \
        --gradient-checkpointing \
        "$@"
else
    echo "Starting single GPU training..."
    python train_beacon_distill.py \
        --model-path "$MODEL_PATH" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
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
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --distill-weight $DISTILL_WEIGHT \
        --ce-weight $CE_WEIGHT \
        --distill-temperature $DISTILL_TEMP \
        --bf16 \
        --train-beacon-only \
        --train-lm-head \
        --gradient-checkpointing \
        "$@"
fi

echo "Training completed!"
