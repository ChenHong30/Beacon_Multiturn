#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/data/hkustgz/model_weight/Qwen3-4B/"
TEACHER_MODEL_PATH="/data/hkustgz/model_weight/Qwen3-4B/"
DATA_PATHS="dataset_multiturn_generated.jsonl"

# ============ GPU分离配置 ============
# 将教师模型和学生模型放在不同GPU上以节省显存
# STUDENT_GPUS: 用于学生模型DDP训练的GPU（决定DDP并行度）
# TEACHER_GPUS: 用于教师模型推理的GPU
# 留空TEACHER_GPUS则使用传统模式（教师和学生在同一GPU上）
STUDENT_GPUS="0,1,2,3"
TEACHER_GPUS=""
# =====================================
OUTPUT_DIR="/data/hkustgz/model_weight/beacon-4B-dynamic-64"
PROCESSED_CACHE_DIR="./runs/dataset_cache_generated"
MAX_LENGTH="4096"
BATCH_SIZE="1"
GRAD_ACCUM="8"
LEARNING_RATE="1e-4"
NUM_EPOCHS="16"
WARMUP_RATIO="0.03"
SAVE_STEPS="1000"
NUM_BEACONS="24"
MAX_BEACON_NUM="64"  # 动态beacon训练：训练时从1到max随机采样beacon数量
BEACON_NUM_CHOICES="24,64"  # 离散beacon数量选择空间，如 "4,8,16"，为空则使用1到max_beacon_num
NUM_SINKS="0"
BEACON_RECON_WEIGHT="1.0"
DISTILL_WEIGHT="1.0"
CE_WEIGHT="0.0"
DISTILL_TEMP="3.0"
# 新增参数
BEACON_ATTN_WEIGHT="0.5"
MIN_BEACON_ATTN="0.1"
HIDDEN_DISTILL_WEIGHT="0.5"
HIDDEN_DISTILL_LAYER="-1"
# 注意力引导蒸馏参数
ATTN_GUIDED_DISTILL_WEIGHT="0.5"
ATTN_GUIDED_LAYERS="-1"
ATTN_GUIDED_TEMPERATURE="3.0"


# 计算学生模型使用的GPU数量（决定DDP并行度）
if [ -n "$STUDENT_GPUS" ]; then
    N_GPUS=$(echo "$STUDENT_GPUS" | tr ',' '\n' | wc -l)
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(nvidia-smi -L | wc -l)
else
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

# 构建教师GPU参数
TEACHER_GPUS_ARG=""
if [ -n "$TEACHER_GPUS" ]; then
    TEACHER_GPUS_ARG="--teacher-gpus $TEACHER_GPUS"
fi

echo "========================================"
echo "Beacon Distillation Training (v2)"
echo "========================================"
echo "Student GPUs (DDP): $STUDENT_GPUS (n=$N_GPUS)"
echo "Teacher GPUs: $TEACHER_GPUS"
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
echo "Max beacon num: $MAX_BEACON_NUM"
echo "Beacon num choices: $BEACON_NUM_CHOICES"
echo "Num sinks: $NUM_SINKS"
echo "Beacon recon weight: $BEACON_RECON_WEIGHT"
echo "Distill weight: $DISTILL_WEIGHT"
echo "CE weight: $CE_WEIGHT"
echo "Distill temperature: $DISTILL_TEMP"
echo "Beacon attn weight: $BEACON_ATTN_WEIGHT"
echo "Hidden distill weight: $HIDDEN_DISTILL_WEIGHT"
echo "Hidden distill layer: $HIDDEN_DISTILL_LAYER"
echo "Min beacon attn: $MIN_BEACON_ATTN"
echo "Attn guided distill weight: $ATTN_GUIDED_DISTILL_WEIGHT"
echo "Attn guided layers: $ATTN_GUIDED_LAYERS"
echo "Attn guided temperature: $ATTN_GUIDED_TEMPERATURE"
echo "========================================"

# 构建可选参数
BEACON_NUM_CHOICES_ARG=""
if [ -n "$BEACON_NUM_CHOICES" ]; then
    BEACON_NUM_CHOICES_ARG="--beacon-num-choices $BEACON_NUM_CHOICES"
fi

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
        --max-beacon-num $MAX_BEACON_NUM \
        $BEACON_NUM_CHOICES_ARG \
        --num-sinks $NUM_SINKS \
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --distill-weight $DISTILL_WEIGHT \
        --ce-weight $CE_WEIGHT \
        --distill-temperature $DISTILL_TEMP \
        --beacon-attn-weight $BEACON_ATTN_WEIGHT \
        --hidden-distill-weight $HIDDEN_DISTILL_WEIGHT \
        --hidden-distill-layer $HIDDEN_DISTILL_LAYER \
        --min-beacon-attn $MIN_BEACON_ATTN \
        --attn-guided-distill-weight $ATTN_GUIDED_DISTILL_WEIGHT \
        --attn-guided-layers "$ATTN_GUIDED_LAYERS" \
        --attn-guided-temperature $ATTN_GUIDED_TEMPERATURE \
        --bf16 \
        --train-beacon-only \
        --gradient-checkpointing \
        $TEACHER_GPUS_ARG \
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
        --max-beacon-num $MAX_BEACON_NUM \
        $BEACON_NUM_CHOICES_ARG \
        --num-sinks $NUM_SINKS \
        --beacon-recon-weight $BEACON_RECON_WEIGHT \
        --distill-weight $DISTILL_WEIGHT \
        --ce-weight $CE_WEIGHT \
        --distill-temperature $DISTILL_TEMP \
        --beacon-attn-weight $BEACON_ATTN_WEIGHT \
        --hidden-distill-weight $HIDDEN_DISTILL_WEIGHT \
        --hidden-distill-layer $HIDDEN_DISTILL_LAYER \
        --min-beacon-attn $MIN_BEACON_ATTN \
        --attn-guided-distill-weight $ATTN_GUIDED_DISTILL_WEIGHT \
        --attn-guided-layers "$ATTN_GUIDED_LAYERS" \
        --attn-guided-temperature $ATTN_GUIDED_TEMPERATURE \
        --bf16 \
        --train-beacon-only \
        --train-lm-head \
        --gradient-checkpointing \
        $TEACHER_GPUS_ARG \
        "$@"
fi

echo "Training completed!"

bash eval/eval.sh multi_if
bash eval/eval.sh mtbench_101
bash eval/eval.sh gsm8k_variant
bash eval/eval.sh coreference_resolution
