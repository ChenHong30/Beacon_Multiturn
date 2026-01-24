
#!/bin/bash

# ------------------------------------------------------------------------------------------
# Basic Configuration
BEACON_MODEL_PATH="/home/hkustgz/Beacon_Multiturn/model_weight/beacon-1.7B-dynamic-64" # Path to the beacon model
BASE_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-1.7B"
CUDA_ID=0,1,2,3 # Comma-separated CUDA device IDs
TASK_TYPE="${1:-safediabench}" # Options: multi_if, mtbench_101, gsm8k_variant, coreference_resolution, mhj, safediabench
MODEL_TYPE="base" # Options: beacon, base
NUM_WORKERS=8 # Number of parallel workers for data loading
if [ "$MODEL_TYPE" = "beacon" ]; then
    LOG_DIR="./logs/${TASK_TYPE}/$(basename "$BEACON_MODEL_PATH")" # Log directory based on task and model
else
    LOG_DIR="./logs/${TASK_TYPE}/$(basename "$BASE_MODEL_PATH")" # Log directory based on task and model
fi
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
NUM_BEACONS=64
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (MTBench-101)
DATA_PATH="./eval/mtbench_101/mtbench101.jsonl"
DATA_NAME=""  # set when DATA_PATH is a directory
CONFIG_PATH="./eval/mtbench_101/mtbench_101_config.json"
MAX_NEW_TOKENS_MTBENCH=2048
TEMPERATURE_MTBENCH=0.7
DO_SAMPLE=true
TOP_P=""
FLUSH_EVERY=1
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (GSM8K Variant)
GSM8K_VARIANT_PATH="./eval/gsm8k_variant/gsm8k_variant_dataset.jsonl"
MAX_INPUT_TOKENS_GSM8K=8192
MAX_NEW_TOKENS_GSM8K=1024
TEMPERATURE_GSM8K=0.7
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (Coreference Resolution)
COREF_PATH="./eval/coreference_resolution/coref_dataset.jsonl"
MAX_INPUT_TOKENS_COREF=8192
MAX_NEW_TOKENS_COREF=256
TEMPERATURE_COREF=0.7
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (MHJ - Multi-turn Human Jailbreaking)
MHJ_PATH="./eval/mhj/mhj_dataset.jsonl"
MHJ_CONFIG_PATH="./eval/mhj/mhj_config.json"
MAX_NEW_TOKENS_MHJ=1024
TEMPERATURE_MHJ=0.0
DO_SAMPLE_MHJ=false
# ------------------------------------------------------------------------------------------
# Task-specific Scripts (SafeDialBench - Fine-Grained Safety Benchmark)
SAFEDIABENCH_PATH="./eval/safediabench/safediabench_dataset.jsonl"
SAFEDIABENCH_CONFIG_PATH="./eval/safediabench/safediabench_config.json"
SAFEDIABENCH_PROMPTS_DIR="./eval/safediabench/judge_prompts"
MAX_NEW_TOKENS_SAFEDIABENCH=1024
TEMPERATURE_SAFEDIABENCH=0.0
DO_SAMPLE_SAFEDIABENCH=false
ENABLE_SAMPLING_SAFEDIABENCH=true
SAMPLE_SIZE_A_SAFEDIABENCH=200
SAMPLE_SIZE_B_SAFEDIABENCH=200
SAMPLE_SEED_SAFEDIABENCH=42
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
    if [ -n "$DATA_NAME" ]; then
        echo "Data Name : $DATA_NAME"
    fi
    echo "Config    : $CONFIG_PATH"
    echo "Max New Tokens: $MAX_NEW_TOKENS_MTBENCH"
    echo "Temperature   : $TEMPERATURE_MTBENCH"
    echo "Do Sample     : $DO_SAMPLE"
    echo "Top P         : ${TOP_P:-'N/A'}"
    echo "Flush Every   : $FLUSH_EVERY"
elif [ "$TASK_TYPE" = "gsm8k_variant" ]; then
    echo "Data Path : $GSM8K_VARIANT_PATH"
    echo "Max Input Tokens: $MAX_INPUT_TOKENS_GSM8K"
    echo "Max New Tokens  : $MAX_NEW_TOKENS_GSM8K"
    echo "Temperature     : $TEMPERATURE_GSM8K"
elif [ "$TASK_TYPE" = "coreference_resolution" ]; then
    echo "Data Path : $COREF_PATH"
    echo "Max Input Tokens: $MAX_INPUT_TOKENS_COREF"
    echo "Max New Tokens  : $MAX_NEW_TOKENS_COREF"
    echo "Temperature     : $TEMPERATURE_COREF"
elif [ "$TASK_TYPE" = "mhj" ]; then
    echo "Data Path : $MHJ_PATH"
    echo "Config    : $MHJ_CONFIG_PATH"
    echo "Max New Tokens: $MAX_NEW_TOKENS_MHJ"
    echo "Temperature   : $TEMPERATURE_MHJ"
    echo "Do Sample     : $DO_SAMPLE_MHJ"
elif [ "$TASK_TYPE" = "safediabench" ]; then
    echo "Data Path : $SAFEDIABENCH_PATH"
    echo "Config    : $SAFEDIABENCH_CONFIG_PATH"
    echo "Prompts Dir   : $SAFEDIABENCH_PROMPTS_DIR"
    echo "Max New Tokens: $MAX_NEW_TOKENS_SAFEDIABENCH"
    echo "Temperature   : $TEMPERATURE_SAFEDIABENCH"
    echo "Do Sample     : $DO_SAMPLE_SAFEDIABENCH"
    echo "Enable Sampling: $ENABLE_SAMPLING_SAFEDIABENCH"
    if [ "$ENABLE_SAMPLING_SAFEDIABENCH" = "true" ]; then
        echo "Sample Size A : $SAMPLE_SIZE_A_SAFEDIABENCH"
        echo "Sample Size B : $SAMPLE_SIZE_B_SAFEDIABENCH"
        echo "Sample Seed   : $SAMPLE_SEED_SAFEDIABENCH"
    fi
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
    NAME_ARG=""
    if [ -n "$DATA_NAME" ]; then
        NAME_ARG="--name=$DATA_NAME"
    fi
    TOP_P_ARG=""
    if [ -n "$TOP_P" ]; then
        TOP_P_ARG="--top_p=$TOP_P"
    fi

    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/mtbench_101/eval_mtbench_101_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --data_path="$DATA_PATH" \
            $NAME_ARG \
            --config_path="$CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --max_new_tokens="$MAX_NEW_TOKENS_MTBENCH" \
            --temperature="$TEMPERATURE_MTBENCH" \
            --do_sample="$DO_SAMPLE" \
            $TOP_P_ARG \
            --flush_every="$FLUSH_EVERY" \
            --num_workers="$NUM_WORKERS"
    else
        python "eval/mtbench_101/eval_mtbench_101_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --data_path="$DATA_PATH" \
            $NAME_ARG \
            --config_path="$CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --max_new_tokens="$MAX_NEW_TOKENS_MTBENCH" \
            --temperature="$TEMPERATURE_MTBENCH" \
            --do_sample="$DO_SAMPLE" \
            $TOP_P_ARG \
            --flush_every="$FLUSH_EVERY" \
            --num_workers="$NUM_WORKERS"
    fi
elif [ "$TASK_TYPE" = "gsm8k_variant" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/gsm8k_variant/eval_gsm8k_variant_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --num_workers="$NUM_WORKERS" \
            --gsm8k_variant_path="$GSM8K_VARIANT_PATH" \
            --max_input_tokens="$MAX_INPUT_TOKENS_GSM8K" \
            --max_new_tokens="$MAX_NEW_TOKENS_GSM8K" \
            --temperature="$TEMPERATURE_GSM8K"
    else
        python "eval/gsm8k_variant/eval_gsm8k_variant_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS" \
            --gsm8k_variant_path="$GSM8K_VARIANT_PATH" \
            --max_input_tokens="$MAX_INPUT_TOKENS_GSM8K" \
            --max_new_tokens="$MAX_NEW_TOKENS_GSM8K" \
            --temperature="$TEMPERATURE_GSM8K"
    fi
elif [ "$TASK_TYPE" = "coreference_resolution" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/coreference_resolution/eval_coreference_resolution_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --num_workers="$NUM_WORKERS" \
            --coref_path="$COREF_PATH" \
            --max_input_tokens="$MAX_INPUT_TOKENS_COREF" \
            --max_new_tokens="$MAX_NEW_TOKENS_COREF" \
            --temperature="$TEMPERATURE_COREF"
    else
        python "eval/coreference_resolution/eval_coreference_resolution_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS" \
            --coref_path="$COREF_PATH" \
            --max_input_tokens="$MAX_INPUT_TOKENS_COREF" \
            --max_new_tokens="$MAX_NEW_TOKENS_COREF" \
            --temperature="$TEMPERATURE_COREF"
    fi
elif [ "$TASK_TYPE" = "mhj" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/mhj/eval_mhj_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --data_path="$MHJ_PATH" \
            --config_path="$MHJ_CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --num_workers="$NUM_WORKERS" \
            --max_new_tokens="$MAX_NEW_TOKENS_MHJ" \
            --temperature="$TEMPERATURE_MHJ" \
            --do_sample="$DO_SAMPLE_MHJ"
    else
        python "eval/mhj/eval_mhj_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --data_path="$MHJ_PATH" \
            --config_path="$MHJ_CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS" \
            --max_new_tokens="$MAX_NEW_TOKENS_MHJ" \
            --temperature="$TEMPERATURE_MHJ" \
            --do_sample="$DO_SAMPLE_MHJ"
    fi
elif [ "$TASK_TYPE" = "safediabench" ]; then
    if [ "$MODEL_TYPE" = "beacon" ]; then
        python "eval/safediabench/eval_safediabench_beacon.py" \
            --model_path="$BEACON_MODEL_PATH" \
            --data_path="$SAFEDIABENCH_PATH" \
            --prompts_dir="$SAFEDIABENCH_PROMPTS_DIR" \
            --config_path="$SAFEDIABENCH_CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$NUM_BEACONS" \
            --num_workers="$NUM_WORKERS" \
            --max_new_tokens="$MAX_NEW_TOKENS_SAFEDIABENCH" \
            --temperature="$TEMPERATURE_SAFEDIABENCH" \
            --do_sample="$DO_SAMPLE_SAFEDIABENCH" \
            $([ "$ENABLE_SAMPLING_SAFEDIABENCH" = "true" ] && echo "--enable-sampling") \
            --sample_size_a="$SAMPLE_SIZE_A_SAFEDIABENCH" \
            --sample_size_b="$SAMPLE_SIZE_B_SAFEDIABENCH" \
            --sample_seed="$SAMPLE_SEED_SAFEDIABENCH"
    else
        python "eval/safediabench/eval_safediabench_base.py" \
            --model_path="$BASE_MODEL_PATH" \
            --data_path="$SAFEDIABENCH_PATH" \
            --prompts_dir="$SAFEDIABENCH_PROMPTS_DIR" \
            --config_path="$SAFEDIABENCH_CONFIG_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_workers="$NUM_WORKERS" \
            --max_new_tokens="$MAX_NEW_TOKENS_SAFEDIABENCH" \
            --temperature="$TEMPERATURE_SAFEDIABENCH" \
            --do_sample="$DO_SAMPLE_SAFEDIABENCH" \
            $([ "$ENABLE_SAMPLING_SAFEDIABENCH" = "true" ] && echo "--enable-sampling") \
            --sample_size_a="$SAMPLE_SIZE_A_SAFEDIABENCH" \
            --sample_size_b="$SAMPLE_SIZE_B_SAFEDIABENCH" \
            --sample_seed="$SAMPLE_SEED_SAFEDIABENCH"
    fi
else
    echo "❌ Unknown TASK_TYPE: $TASK_TYPE. Please set to one of: multi_if, mtbench_101, gsm8k_variant, coreference_resolution, mhj, safediabench."
    exit 1
fi


