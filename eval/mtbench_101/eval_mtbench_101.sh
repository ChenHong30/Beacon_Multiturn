#!/bin/bash

IS_BEACON=true
BEACON_MODEL_PATH="/home/hkustgz/Beacon_Multiturn/model_weight/8_beacon_0_sink_distill_generated_dual_attn_temp_2"
BASE_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B"
CUDA_ID=0
LOG_DIR="./logs/mtbench_101"

DATA_PATH="./eval/mtbench_101/mtbench101.jsonl"
DATA_NAME=""  # set when DATA_PATH is a directory
CONFIG_PATH="eval/mtbench_101/mtbench_101_config.json"

NUM_SINKS=0
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
DO_SAMPLE=true
TOP_P=""
FLUSH_EVERY=1

BEACON_SCRIPT="eval/mtbench_101/eval_mtbench_101_beacon.py"
BASE_SCRIPT="eval/mtbench_101/eval_mtbench_101_base.py"

echo "🚀 Evaluating MTBench 101..."
echo "----------------------------------------------------"
if [ "$IS_BEACON" = "true" ]; then
    echo "Model Path: $BEACON_MODEL_PATH"
else
    echo "Model Path: $BASE_MODEL_PATH"
fi
echo "CUDA ID   : $CUDA_ID"
echo "Log Dir   : $LOG_DIR"
echo "Data Path : $DATA_PATH"
if [ -n "$DATA_NAME" ]; then
    echo "Data Name : $DATA_NAME"
fi
echo "Config    : $CONFIG_PATH"
if [ "$IS_BEACON" = "true" ]; then
    echo "Mode      : BEACON (Num Sinks: $NUM_SINKS)"
else
    echo "Mode      : BASE"
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

NAME_ARG=""
if [ -n "$DATA_NAME" ]; then
    NAME_ARG="--name=$DATA_NAME"
fi

TOP_P_ARG=""
if [ -n "$TOP_P" ]; then
    TOP_P_ARG="--top_p=$TOP_P"
fi

if [ "$IS_BEACON" = "true" ]; then
    echo "▶️   Running Beacon Evaluation script: $BEACON_SCRIPT"
    python "$BEACON_SCRIPT" \
        --model_path="$BEACON_MODEL_PATH" \
        --data_path="$DATA_PATH" \
        $NAME_ARG \
        --config_path="$CONFIG_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --num_sinks="$NUM_SINKS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --temperature="$TEMPERATURE" \
        --do_sample="$DO_SAMPLE" \
        $TOP_P_ARG \
        --flush_every="$FLUSH_EVERY"
else
    echo "▶️   Running Base Evaluation script: $BASE_SCRIPT"
    python "$BASE_SCRIPT" \
        --model_path="$BASE_MODEL_PATH" \
        --data_path="$DATA_PATH" \
        $NAME_ARG \
        --config_path="$CONFIG_PATH" \
        --cuda_ids="$CUDA_ID" \
        --log_dir="$LOG_DIR" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --temperature="$TEMPERATURE" \
        --do_sample="$DO_SAMPLE" \
        $TOP_P_ARG \
        --flush_every="$FLUSH_EVERY"
fi
