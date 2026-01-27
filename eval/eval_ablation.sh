#!/bin/bash

# 1. 基础配置
MODELS=(
    # "/hpc2hdd/home/hchen763/jhaidata/local_model/beacon-0.6B-ablation-fixed-k"
    "/home/hkustgz/Beacon_Multiturn/model_weight/beacon-0.6B-dynamic-64"
)
K_VALUES=(10 20 30 40 50 60)

CUDA_ID=0,1
TASK_TYPE="coreference_resolution"
NUM_WORKERS=8
NUM_SINKS=0

# Coreference Resolution 专用路径与参数
COREF_PATH="./eval/coreference_resolution/coref_dataset.jsonl"
MAX_INPUT_TOKENS_COREF=8192
MAX_NEW_TOKENS_COREF=256
TEMPERATURE_COREF=0.7

# 2. 开始循环
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    for K in "${K_VALUES[@]}"; do
        # 动态创建日志目录，区分模型和 k 值
        LOG_DIR="./logs/${TASK_TYPE}/${MODEL_NAME}/k${K}"
        mkdir -p "$LOG_DIR"
        
        echo "----------------------------------------------------------------"
        echo "🚀 Running: $MODEL_NAME with NUM_BEACONS=$K"
        echo "📂 Log Dir: $LOG_DIR"
        
        # 执行 Python 评测脚本
        python "eval/coreference_resolution/eval_coreference_resolution_beacon.py" \
            --model_path="$MODEL_PATH" \
            --cuda_ids="$CUDA_ID" \
            --log_dir="$LOG_DIR" \
            --num_sinks="$NUM_SINKS" \
            --num_beacons="$K" \
            --num_workers="$NUM_WORKERS" \
            --coref_path="$COREF_PATH" \
            --max_input_tokens="$MAX_INPUT_TOKENS_COREF" \
            --max_new_tokens="$MAX_NEW_TOKENS_COREF" \
            --temperature="$TEMPERATURE_COREF"
            
        echo "✅ Finished k=$K for $MODEL_NAME"
    done
done

echo "🎉 All ablation tasks completed!"