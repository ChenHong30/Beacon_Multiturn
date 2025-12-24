#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exit immediately if a command exits with a non-zero status
set -e

# Obtain the directory of the script and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parameters
MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B" # The path of base model, it can be a local path or a model name from Hugging Face
DATA_PATHS="ultrachat-40k-le3turns.jsonl" # The path(s) of training data, can be multiple paths separated by space
OUTPUT_DIR="/data/hkustgz/model_weight/8_beacon_4_sink_0.6B_beacon_CE_1e-4" # The output directory to save trained model and logs
PROCESSED_CACHE_DIR="./runs/dataset_cache_40k" # The directory to cache processed datasets
MAX_LENGTH="4096" # The maximum sequence length
BATCH_SIZE="1" # The batch size per device
GRAD_ACCUM="2" # The number of gradient accumulation steps
LEARNING_RATE="1e-4" # The learning rate
NUM_EPOCHS="8" # The number of training epochs
WARMUP_RATIO="0.03" # The warmup ratio
SAVE_STEPS="500" # The number of steps between saving model checkpoints
NUM_BEACONS="8" # The number of beacons per conversation turn
NUM_SINKS="4" # The number of sinks per conversation turn
BEACON_RECON_WEIGHT="0.3" # Weight for auxiliary beacon reconstruction loss
BEACON_CE_WEIGHT="0.3" # Weight for auxiliary beacon token prediction loss
BEACON_LABEL_MODE="chunk_last" # chunk_last or chunk_mid

# Determine the number of GPUs available
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
echo "Beacon recon weight: $BEACON_RECON_WEIGHT"
echo "Beacon CE weight: $BEACON_CE_WEIGHT"
echo "Beacon label mode: $BEACON_LABEL_MODE"
echo "========================================"

if [ "$N_GPUS" -gt 1 ]; then
    echo "Starting distributed training with $N_GPUS GPUs..."
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
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --beacon-ce-weight $BEACON_CE_WEIGHT \
        --beacon-label-mode $BEACON_LABEL_MODE \
        --bf16 \
        --train-beacon-only \
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
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --beacon-ce-weight $BEACON_CE_WEIGHT \
        --beacon-label-mode $BEACON_LABEL_MODE \
        --bf16 \
        --train-beacon-only \
        --train-lm-head \
        --gradient-checkpointing \
        "$@"
fi

echo "Training completed!"
