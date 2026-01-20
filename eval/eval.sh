
#!/bin/bash

# ------------------------------------------------------------------------------------------
# Basic Configuration
BEACON_MODEL_PATH="/home/hkustgz/Beacon_Multiturn/model_weight/beacon-0.6B-dynamic-64" # Path to the beacon model
BASE_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-0.6B"
CUDA_ID=0,1,2,3 # Comma-separated CUDA device IDs
TASK_TYPE="multi_if" # Options: multi_if, mtbench_101, gsm8k
MODEL_TYPE="beacon" # Options: beacon, base
NUM_WORKERS=32 # Number of parallel workers for data loading
LOG_DIR="./logs/${TASK_TYPE}/$(basename "$BEACON_MODEL_PATH")" # Log directory based on task and model
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Unable to create log directory: $LOG_DIR"
        exit 1
    fi
    echo "✅ Created log directory: $LOG_DIR"
fi
# ------------------------------------------------------------------------------------------
# Beacon Configuration
NUM_SINKS=0
NUM_BEACONS=25
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (MTBench-101)
DATA_PATH="./eval/mtbench_101/mtbench101.jsonl"
OPENAI_CONFIG_PATH="./eval/mtbench_101/mtbench_101_config.json"
MAX_NEW_TOKENS=2048
TEMPERATURE_MTBENCH=0.7
DO_SAMPLE=true
TOP_P=""
FLUSH_EVERY=1
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (GSM8K)
HISTORY_MAX_TURNS=6
MAX_INPUT_TOKENS=4096
MAX_NEW_TOKENS_GSM8K=1024
ULTRACHAT_PATH="./ultrachat-200k.jsonl"
GSM8K_SPLIT="test"
TEMPERATURE_GSM8K=0.7
# ------------------------------------------------------------------------------------------
# Evaluation Execution
echo "🚀 Evaluating $TASK_TYPE with $MODEL_TYPE model"
echo "Model Path: $([ "$MODEL_TYPE" = "beacon" ] && echo "$BEACON_MODEL_PATH" || echo "$BASE_MODEL_PATH")"
echo "Num Sinks : $NUM_SINKS"
echo "Num Beacons: $NUM_BEACONS"
echo "CUDA ID   : $CUDA_ID"
echo "Workers   : $NUM_WORKERS"
echo "Log Dir   : $LOG_DIR"
if [ "$TASK_TYPE" = "mtbench_101" ]; then
    echo "Data Path : $DATA_PATH"
    echo "Config    : $OPENAI_CONFIG_PATH"
    echo "Max New Tokens: $MAX_NEW_TOKENS"
    echo "Temperature   : $TEMPERATURE_MTBENCH"
    echo "Do Sample     : $DO_SAMPLE"
    echo "Top P         : ${TOP_P:-'N/A'}"
    echo "Flush Every   : $FLUSH_EVERY"
elif [ "$TASK_TYPE" = "gsm8k" ]; then
    echo "Ultrachat Path: $ULTRACHAT_PATH"
    echo "Max Input Tokens: $MAX_INPUT_TOKENS"
    echo "Max New Tokens  : $MAX_NEW_TOKENS_GSM8K"
    echo "Split     : $GSM8K_SPLIT"
    echo "Max Turns     : $HISTORY_MAX_TURNS"
    echo "Temp      : $TEMPERATURE_GSM8K"
fi

if [ "$TASK_TYPE" = "multi_if" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/multi_if/eval_multi_if_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --num_workers="$NUM_WORKERS"
    else
        python "eval/multi_if/eval_multi_if_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS"
    fi
elif [ "$TASK_TYPE" = "mtbench_101" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/mtbench_101/eval_mtbench_101_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --data_path="$DATA_PATH" \
            --config_path="$OPENAI_CONFIG_PATH" \
            --cuda_id="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --max_new_tokens="$MAX_NEW_TOKENS" \
            --temperature="$TEMPERATURE" \
            --do_sample="$DO_SAMPLE" \
            --top_p="$TOP_P" \
            --flush_every="$FLUSH_EVERY" \
            --num_workers="$NUM_WORKERS"
    else
        python "eval/mtbench_101/eval_mtbench_101_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --data_path="$DATA_PATH" \
            --config_path="$OPENAI_CONFIG_PATH" \
            --cuda_id="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --max_new_tokens="$MAX_NEW_TOKENS" \
            --temperature="$TEMPERATURE" \
            --do_sample="$DO_SAMPLE" \
            --top_p="$TOP_P" \
            --flush_every="$FLUSH_EVERY" \
            --num_workers="$NUM_WORKERS"
    fi
elif [ "$TASK_TYPE" = "gsm8k" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/gsm8k_interference/eval_gsm8k_interference_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_workers="$NUM_WORKERS" \
            --ultrachat_path="$ULTRACHAT_PATH" \
            --history_max_turns="$HISTORY_MAX_TURNS" \
            --max_input_tokens="$MAX_INPUT_TOKENS" \
            --max_new_tokens="$MAX_NEW_TOKENS_GSM8K" \
            --gsm8k_split="$GSM8K_SPLIT" \
            --temperature="$TEMPERATURE_GSM8K"
    else
        python "eval/gsm8k_interference/eval_gsm8k_interference_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS" \
            --ultrachat_path="$ULTRACHAT_PATH" \
            --history_max_turns="$HISTORY_MAX_TURNS" \
            --max_input_tokens="$MAX_INPUT_TOKENS" \
            --max_new_tokens="$MAX_NEW_TOKENS_GSM8K" \
            --gsm8k_split="$GSM8K_SPLIT" \
            --temperature="$TEMPERATURE_GSM8K"
    fi
else
    echo "❌ Unknown TASK_TYPE: $TASK_TYPE. Please set to one of: multi_if, mtbench_101, gsm8k."
    exit 1
fi


