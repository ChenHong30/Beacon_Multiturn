#!/bin/bash

IS_BEACON=true
BEACON_MODEL_PATH="/data/hkustgz/model_weight/8_beacon_0_sink_distill_v2_turn_embedding"
BASE_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B"
CUDA_ID=0,1,2,3
LOG_DIR="/home/hkustgz/Beacon_Multiturn/logs/gsm8k_interference/8_beacon_0_sink_distill_v2_turn_embedding"

# GSM8K Specific Config
HISTORY_MAX_TURNS=6
MAX_INPUT_TOKENS=4096
MAX_NEW_TOKENS=1024
ULTRACHAT_PATH="/home/hkustgz/Beacon_Multiturn/ultrachat-200k.jsonl"
GSM8K_SPLIT="test"
TEMPERATURE=0.7

NUM_SINKS=0
NUM_WORKERS=32
BEACON_SCRIPT="eval/gsm8k_interference/eval_gsm8k_interference_beacon.py"
BASE_SCRIPT="eval/gsm8k_interference/eval_gsm8k_interference_base.py"

echo "🚀 Evaluating GSM8K Interference..."
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
echo "Split     : $GSM8K_SPLIT"
echo "Turns     : $HISTORY_MAX_TURNS"
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
        --num_sinks="$NUM_SINKS" \
        --num_workers="$NUM_WORKERS" \
        --ultrachat_path="$ULTRACHAT_PATH" \
        --history_max_turns="$HISTORY_MAX_TURNS" \
        --max_input_tokens="$MAX_INPUT_TOKENS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --gsm8k_split="$GSM8K_SPLIT" \
        --temperature="$TEMPERATURE"

else
    # ------------------------------------
    # Case 2: Run Base Script
    # ------------------------------------
    echo "▶️   Running Base Evaluation script: $BASE_SCRIPT"
    
    python "$BASE_SCRIPT" \
        --model_path="$BASE_MODEL_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --num_workers="$NUM_WORKERS" \
        --ultrachat_path="$ULTRACHAT_PATH" \
        --history_max_turns="$HISTORY_MAX_TURNS" \
        --max_input_tokens="$MAX_INPUT_TOKENS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --gsm8k_split="$GSM8K_SPLIT" \
        --temperature="$TEMPERATURE"
fi
