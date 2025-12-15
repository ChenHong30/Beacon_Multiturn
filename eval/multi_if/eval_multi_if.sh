#!/bin/bash

IS_BEACON=true
BEACON_MODEL_PATH="/data/hkustgz/model_weight/16_beacon_4_sink"
BASE_MODEL_PATH="/home/hkustgz/model_weights/Qwen3-8B"
CUDA_ID=0,1,2,3
LOG_DIR="/home/hkustgz/Beacon_Multiturn/logs"
NUM_SINKS=1
BEACON_SCRIPT="eval/multi_if/eval_multi_if_beacon.py"
BASE_SCRIPT="eval/multi_if/eval_multi_if_base.py"

echo "🚀 Evaluating Multi-IF..."
echo "----------------------------------------------------"
if [ "$IS_BEACON" = "true" ]; then
    echo "Model Path: $BEACON_MODEL_PATH"
    MODEL_PATH="$BEACON_MODEL_PATH"
else
    echo "Model Path: $BASE_MODEL_PATH"
    MODEL_PATH="$BASE_MODEL_PATH"
fi
echo "CUDA ID   : $CUDA_ID"
echo "Log Dir   : $LOG_DIR"
if [ "$IS_BEACON" = "true" ]; then
    echo "Mode      : BEACON (Num Sinks: $NUM_SINKS)"
else
    echo "Mode      : BASE (Standard Model)"
fi
echo "----------------------------------------------------"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Unable to create log directory: $LOG_DIR"
        exit 1
    fi
    echo "✅ Created log directory: $LOG_DIR"
fi

if [ "$IS_BEACON" = "true" ]; then
    # ------------------------------------
    # Case 1: Run Beacon Script
    # ------------------------------------
    echo "▶️   Running Beacon Evaluation script: $BEACON_SCRIPT"
    
    python "$BEACON_SCRIPT" \
        --model_path="$BEACON_MODEL_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --num_sinks="$NUM_SINKS"

else
    # ------------------------------------
    # Case 2: Run Base Script
    # ------------------------------------
    echo "▶️   Running Base Evaluation script: $BASE_SCRIPT"
    # Base script only supports single GPU; take the first id if a list is provided (e.g. "0,1,2,3").
    FIRST_CUDA_ID="$(echo "$CUDA_ID" | sed 's/[ ,].*$//')"
    
    python "$BASE_SCRIPT" \
        --model_path="$BASE_MODEL_PATH" \
        --cuda_id="$FIRST_CUDA_ID" \
        --log_dir="$LOG_DIR"
fi
