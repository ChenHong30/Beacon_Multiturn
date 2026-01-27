#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ==============================
# 可通过环境变量覆盖的基础配置
# ==============================
MODEL_PATH="${MODEL_PATH:-/data/hkustgz/model_weight/Qwen3-0.6B/}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MODEL_PATH}"
DATA_PATHS="${DATA_PATHS:-dataset_multiturn_generated.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/beacon_embedding_ablations}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-$PROJECT_ROOT/runs/dataset_cache_ablations}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/coreference_resolution/beacon_embedding_ablations}"

# 训练超参（默认对齐现有 train_beacon_distill.sh）
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SEED="${SEED:-42}"

NUM_BEACONS="${NUM_BEACONS:-24}"
MAX_BEACON_NUM="${MAX_BEACON_NUM:-64}"
BEACON_NUM_CHOICES="${BEACON_NUM_CHOICES:-24,64}"
NUM_SINKS="${NUM_SINKS:-0}"

BEACON_RECON_WEIGHT="${BEACON_RECON_WEIGHT:-1.0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
CE_WEIGHT="${CE_WEIGHT:-0.0}"
DISTILL_TEMP="${DISTILL_TEMP:-3.0}"

BEACON_ATTN_WEIGHT="${BEACON_ATTN_WEIGHT:-0.5}"
MIN_BEACON_ATTN="${MIN_BEACON_ATTN:-0.1}"
HIDDEN_DISTILL_WEIGHT="${HIDDEN_DISTILL_WEIGHT:-0.5}"
HIDDEN_DISTILL_LAYER="${HIDDEN_DISTILL_LAYER:--1}"

ATTN_GUIDED_DISTILL_WEIGHT="${ATTN_GUIDED_DISTILL_WEIGHT:-0.5}"
ATTN_GUIDED_LAYERS="${ATTN_GUIDED_LAYERS:--1}"
ATTN_GUIDED_TEMPERATURE="${ATTN_GUIDED_TEMPERATURE:-3.0}"

# 硬件/并行配置
STUDENT_GPUS="${STUDENT_GPUS:-0,1,2,3}"
TEACHER_GPUS="${TEACHER_GPUS:-}"
TORCH_MASTER_PORT="${TORCH_MASTER_PORT:-29511}"

# Coreference Resolution 评测配置
EVAL_CUDA_IDS="${EVAL_CUDA_IDS:-$STUDENT_GPUS}"
NUM_WORKERS="${NUM_WORKERS:-16}"
COREF_PATH="${COREF_PATH:-$PROJECT_ROOT/eval/coreference_resolution/coref_dataset.jsonl}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 解析 data paths（允许空格分隔的多个文件）
read -r -a DATA_PATHS_ARR <<< "$DATA_PATHS"

# 计算学生模型使用的 GPU 数量（决定 DDP 并行度）
if [ -n "$STUDENT_GPUS" ]; then
  N_GPUS=$(echo "$STUDENT_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
else
  N_GPUS=1
fi

# 可选参数构造（用数组避免引号问题）
BEACON_NUM_CHOICES_ARG=()
if [ -n "$BEACON_NUM_CHOICES" ]; then
  BEACON_NUM_CHOICES_ARG=(--beacon-num-choices "$BEACON_NUM_CHOICES")
fi

TEACHER_GPUS_ARG=()
if [ -n "$TEACHER_GPUS" ]; then
  TEACHER_GPUS_ARG=(--teacher-gpus "$TEACHER_GPUS")
fi

mkdir -p "$OUTPUT_ROOT" "$PROCESSED_CACHE_ROOT" "$LOG_ROOT"

ABLATION_NAMES=(
  "no_turn_embedding"
  "no_num_beacons_embedding"
  "no_semantic_init"
)

ABLATION_FLAGS=(
  "--disable-turn-embedding"
  "--disable-num-beacons-embedding"
  "--disable-semantic-init"
)

echo "========================================"
echo "Beacon Embedding Ablations (train+eval)"
echo "Project root : $PROJECT_ROOT"
echo "Model path   : $MODEL_PATH"
echo "Teacher path : $TEACHER_MODEL_PATH"
echo "Data paths   : ${DATA_PATHS_ARR[*]}"
echo "Output root  : $OUTPUT_ROOT"
echo "Log root     : $LOG_ROOT"
echo "Student GPUs : $STUDENT_GPUS (n=$N_GPUS)"
echo "Eval GPUs    : $EVAL_CUDA_IDS"
echo "Coref path   : $COREF_PATH"
echo "Num beacons  : $NUM_BEACONS (max=$MAX_BEACON_NUM; choices=$BEACON_NUM_CHOICES)"
echo "Num sinks    : $NUM_SINKS"
echo "========================================"

run_training() {
  local ablation_name="$1"
  local ablation_flag="$2"
  local output_dir="$3"
  local cache_dir="$4"
  local master_port="$5"

  echo ""
  echo "----------------------------------------"
  echo "[Train] $ablation_name"
  echo "Output dir : $output_dir"
  echo "Cache dir  : $cache_dir"
  echo "Flag       : $ablation_flag"
  echo "Master port: $master_port"
  echo "----------------------------------------"

  if [ "$N_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$STUDENT_GPUS" \
      torchrun \
        --nproc_per_node="$N_GPUS" \
        --master_port="$master_port" \
        train_beacon_distill.py \
        --model-path "$MODEL_PATH" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
        --data-paths "${DATA_PATHS_ARR[@]}" \
        --output-dir "$output_dir" \
        --processed-cache-dir "$cache_dir" \
        --max-length "$MAX_LENGTH" \
        --per-device-train-batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LEARNING_RATE" \
        --num-epochs "$NUM_EPOCHS" \
        --warmup-ratio "$WARMUP_RATIO" \
        --save-steps "$SAVE_STEPS" \
        --seed "$SEED" \
        --num-beacons "$NUM_BEACONS" \
        --max-beacon-num "$MAX_BEACON_NUM" \
        "${BEACON_NUM_CHOICES_ARG[@]}" \
        --num-sinks "$NUM_SINKS" \
        --beacon-recon-weight "$BEACON_RECON_WEIGHT" \
        --distill-weight "$DISTILL_WEIGHT" \
        --ce-weight "$CE_WEIGHT" \
        --distill-temperature "$DISTILL_TEMP" \
        --beacon-attn-weight "$BEACON_ATTN_WEIGHT" \
        --min-beacon-attn "$MIN_BEACON_ATTN" \
        --hidden-distill-weight "$HIDDEN_DISTILL_WEIGHT" \
        --hidden-distill-layer "$HIDDEN_DISTILL_LAYER" \
        --attn-guided-distill-weight "$ATTN_GUIDED_DISTILL_WEIGHT" \
        --attn-guided-layers "$ATTN_GUIDED_LAYERS" \
        --attn-guided-temperature "$ATTN_GUIDED_TEMPERATURE" \
        "${TEACHER_GPUS_ARG[@]}" \
        "$ablation_flag"
  else
    CUDA_VISIBLE_DEVICES="$STUDENT_GPUS" \
      python train_beacon_distill.py \
        --model-path "$MODEL_PATH" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
        --data-paths "${DATA_PATHS_ARR[@]}" \
        --output-dir "$output_dir" \
        --processed-cache-dir "$cache_dir" \
        --max-length "$MAX_LENGTH" \
        --per-device-train-batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LEARNING_RATE" \
        --num-epochs "$NUM_EPOCHS" \
        --warmup-ratio "$WARMUP_RATIO" \
        --save-steps "$SAVE_STEPS" \
        --seed "$SEED" \
        --num-beacons "$NUM_BEACONS" \
        --max-beacon-num "$MAX_BEACON_NUM" \
        "${BEACON_NUM_CHOICES_ARG[@]}" \
        --num-sinks "$NUM_SINKS" \
        --beacon-recon-weight "$BEACON_RECON_WEIGHT" \
        --distill-weight "$DISTILL_WEIGHT" \
        --ce-weight "$CE_WEIGHT" \
        --distill-temperature "$DISTILL_TEMP" \
        --beacon-attn-weight "$BEACON_ATTN_WEIGHT" \
        --min-beacon-attn "$MIN_BEACON_ATTN" \
        --hidden-distill-weight "$HIDDEN_DISTILL_WEIGHT" \
        --hidden-distill-layer "$HIDDEN_DISTILL_LAYER" \
        --attn-guided-distill-weight "$ATTN_GUIDED_DISTILL_WEIGHT" \
        --attn-guided-layers "$ATTN_GUIDED_LAYERS" \
        --attn-guided-temperature "$ATTN_GUIDED_TEMPERATURE" \
        "${TEACHER_GPUS_ARG[@]}" \
        "$ablation_flag"
  fi
}

run_coref_eval() {
  local ablation_name="$1"
  local model_path="$2"
  local log_dir="$3"

  mkdir -p "$log_dir"

  echo ""
  echo "----------------------------------------"
  echo "[Eval: Coref] $ablation_name"
  echo "Model path : $model_path"
  echo "Log dir    : $log_dir"
  echo "CUDA IDs   : $EVAL_CUDA_IDS"
  echo "Workers    : $NUM_WORKERS"
  echo "----------------------------------------"

  python eval/coreference_resolution/eval_coreference_resolution_beacon.py \
    --model_path="$model_path" \
    --cuda_ids="$EVAL_CUDA_IDS" \
    --log_dir="$log_dir" \
    --num_sinks="$NUM_SINKS" \
    --num_beacons="$NUM_BEACONS" \
    --num_workers="$NUM_WORKERS" \
    --coref_path="$COREF_PATH" \
    --max_input_tokens="$MAX_INPUT_TOKENS" \
    --max_new_tokens="$MAX_NEW_TOKENS" \
    --temperature="$TEMPERATURE"
}

for i in "${!ABLATION_NAMES[@]}"; do
  name="${ABLATION_NAMES[$i]}"
  flag="${ABLATION_FLAGS[$i]}"

  output_dir="$OUTPUT_ROOT/$name"
  cache_dir="$PROCESSED_CACHE_ROOT/$name"
  log_dir="$LOG_ROOT/$name"

  # 为每个实验使用不同的 master_port，避免残留进程导致端口冲突
  master_port=$((TORCH_MASTER_PORT + i))

  run_training "$name" "$flag" "$output_dir" "$cache_dir" "$master_port"
  run_coref_eval "$name" "$output_dir" "$log_dir"
done

echo ""
echo "All ablations completed."
echo "Outputs:"
echo "- checkpoints: $OUTPUT_ROOT/<ablation_name>"
echo "- coref logs : $LOG_ROOT/<ablation_name>"

