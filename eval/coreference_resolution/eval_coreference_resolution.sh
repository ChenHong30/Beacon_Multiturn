#!/bin/bash

IS_BEACON=false
BEACON_MODEL_PATH="/data/hkustgz/model_weight/8_beacon_0_sink_distill_v2_attnn_guided"
BASE_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B"
CUDA_ID=0,1,2,3
LOG_DIR="/home/hkustgz/Beacon_Multiturn/logs/coreference_resolution/8_beacon_0_sink_distill_v2_attnn_guided"

COREF_PATH="/home/hkustgz/Beacon_Multiturn/eval/coreference_resolution/coref_dataset.jsonl"
MAX_INPUT_TOKENS=8192
MAX_NEW_TOKENS=256
TEMPERATURE=0.7

NUM_SINKS=0
NUM_WORKERS=16
BEACON_SCRIPT="eval/coreference_resolution/eval_coreference_resolution_beacon.py"
BASE_SCRIPT="eval/coreference_resolution/eval_coreference_resolution_base.py"

echo "Running Coreference Resolution evaluation..."
echo "----------------------------------------------------"
if [ "$IS_BEACON" = "true" ]; then
    echo "Model Path: $BEACON_MODEL_PATH"
    MODEL_PATH="$BEACON_MODEL_PATH"
else
    echo "Model Path: $BASE_MODEL_PATH"
    MODEL_PATH="$BASE_MODEL_PATH"
fi
echo "CUDA ID   : $CUDA_ID"
echo "Workers   : $NUM_WORKERS"
echo "Log Dir   : $LOG_DIR"
echo "Data Path : $COREF_PATH"
echo "Temp      : $TEMPERATURE"
if [ "$IS_BEACON" = "true" ]; then
    echo "Mode      : BEACON (Num Sinks: $NUM_SINKS)"
else
    echo "Mode      : BASE (Standard Model)"
fi
echo "----------------------------------------------------"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
        echo "Unable to create log directory: $LOG_DIR"
        exit 1
    fi
    echo "Created log directory: $LOG_DIR"
fi

if [ "$IS_BEACON" = "true" ]; then
    echo "Running Beacon Evaluation script: $BEACON_SCRIPT"
    python "$BEACON_SCRIPT" \
        --model_path="$BEACON_MODEL_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --num_sinks="$NUM_SINKS" \
        --num_workers="$NUM_WORKERS" \
        --coref_path="$COREF_PATH" \
        --max_input_tokens="$MAX_INPUT_TOKENS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --temperature="$TEMPERATURE"
else
    echo "Running Base Evaluation script: $BASE_SCRIPT"
    python "$BASE_SCRIPT" \
        --model_path="$BASE_MODEL_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --num_workers="$NUM_WORKERS" \
        --coref_path="$COREF_PATH" \
        --max_input_tokens="$MAX_INPUT_TOKENS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --temperature="$TEMPERATURE"
fi
