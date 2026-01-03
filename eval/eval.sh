#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -------------------------
# Global params
# -------------------------
# Task selector: gsm8k_interference | mtbench_101 | multi_if
TASK="${TASK:-multi_if}"

# Mode selector: base | beacon | both
MODE="${MODE:-beacon}"

# Shared base model path for all tasks.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/hkustgz/model_weight/Qwen3-0.6B}"
# Shared beacon model path for all tasks.
BEACON_MODEL_PATH="${BEACON_MODEL_PATH:-/data/hkustgz/model_weight/16_beacon_0_sink_distill_v2_turn_embedding/}"
# Shared beacon num_sinks for all tasks.
NUM_SINKS="${NUM_SINKS:-0}"

# Derived model names (last path segment, used in log paths).
BASE_MODEL_NAME="$(basename "${BASE_MODEL_PATH%/}")"
BEACON_MODEL_NAME="$(basename "${BEACON_MODEL_PATH%/}")"
# Log model name (defaults to beacon model name).
LOG_MODEL_NAME="${LOG_MODEL_NAME:-${BEACON_MODEL_NAME}}"

# CUDA GPU ids (comma-separated). Use "0" for single card, "0,1,2,3" for 4 GPUs.
CUDA_IDS="${CUDA_IDS:-0,1,2,3}"
if [ -z "${CUDA_IDS}" ]; then
    CUDA_IDS="0"
fi

# Rank detection for logging (only rank 0 prints).
RANK_ID="${RANK:-}"
if [ -z "${RANK_ID}" ] && [ -n "${LOCAL_RANK:-}" ]; then
    RANK_ID="${LOCAL_RANK}"
fi
if [ -z "${RANK_ID}" ] && [ -n "${SLURM_PROCID:-}" ]; then
    RANK_ID="${SLURM_PROCID}"
fi
if [ -z "${RANK_ID}" ] && [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
    RANK_ID="${OMPI_COMM_WORLD_RANK}"
fi
if [ -z "${RANK_ID}" ] && [ -n "${MPI_RANK:-}" ]; then
    RANK_ID="${MPI_RANK}"
fi
if [ -z "${RANK_ID}" ] && [ -n "${PMI_RANK:-}" ]; then
    RANK_ID="${PMI_RANK}"
fi
if [ -z "${RANK_ID}" ]; then
    RANK_ID="0"
fi

LOG_MAIN_RANK=1
if [ "${RANK_ID}" != "0" ]; then
    LOG_MAIN_RANK=0
fi

# -------------------------
# GSM8K Interference params
# -------------------------
# Optional local GSM8K dataset path (.json/.jsonl). Leave empty to use datasets.
GSM8K_PATH="${GSM8K_PATH:-}"
# UltraChat history file path.
GSM8K_ULTRACHAT_PATH="${GSM8K_ULTRACHAT_PATH:-${PROJECT_ROOT}/ultrachat-200k.jsonl}"
# Output log directory for GSM8K interference (logs/task_name/model_name).
GSM8K_LOG_DIR="${GSM8K_LOG_DIR:-${PROJECT_ROOT}/logs/gsm8k_interference/${LOG_MODEL_NAME}}"
# Optional run tag appended to output filename.
GSM8K_RUN_TAG="${GSM8K_RUN_TAG:-}"
# Random seed for history sampling.
GSM8K_SEED="${GSM8K_SEED:-42}"
# Max samples to evaluate (empty means all).
GSM8K_MAX_SAMPLES="${GSM8K_MAX_SAMPLES:-}"
# Max history turns to include.
GSM8K_HISTORY_MAX_TURNS="${GSM8K_HISTORY_MAX_TURNS:-10}"
# Max input tokens for truncation.
GSM8K_MAX_INPUT_TOKENS="${GSM8K_MAX_INPUT_TOKENS:-4096}"
# Max new tokens to generate.
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-4096}"
# Sampling temperature.
GSM8K_TEMPERATURE="${GSM8K_TEMPERATURE:-0.7}"
# Top-p sampling cutoff.
GSM8K_TOP_P="${GSM8K_TOP_P:-0.8}"
# Enable model thinking mode if supported.
GSM8K_ENABLE_THINKING="${GSM8K_ENABLE_THINKING:-false}"

# -------------------------
# MTBench_101 params
# -------------------------
# Output log directory for MTBench_101 (logs/task_name/model_name).
MTBENCH_LOG_DIR="${MTBENCH_LOG_DIR:-${PROJECT_ROOT}/logs/mtbench_101/${LOG_MODEL_NAME}}"
# MTBench_101 data path (jsonl file or directory).
MTBENCH_DATA_PATH="${MTBENCH_DATA_PATH:-${PROJECT_ROOT}/eval/mtbench_101/mtbench101.jsonl}"
# Dataset name when data path is a directory.
MTBENCH_DATA_NAME="${MTBENCH_DATA_NAME:-}"
# JSON config with openai_api_key/openai_base_url/judge_model.
MTBENCH_CONFIG_PATH="${MTBENCH_CONFIG_PATH:-${PROJECT_ROOT}/eval/mtbench_101/mtbench_101_config.json}"
# Max new tokens for MTBench generation.
MTBENCH_MAX_NEW_TOKENS="${MTBENCH_MAX_NEW_TOKENS:-2048}"
# Sampling temperature for MTBench generation.
MTBENCH_TEMPERATURE="${MTBENCH_TEMPERATURE:-0.7}"
# Whether to sample for MTBench generation.
MTBENCH_DO_SAMPLE="${MTBENCH_DO_SAMPLE:-true}"
# Top-p sampling cutoff for MTBench (empty to skip).
MTBENCH_TOP_P="${MTBENCH_TOP_P:-}"
# Flush interval for partial outputs.
MTBENCH_FLUSH_EVERY="${MTBENCH_FLUSH_EVERY:-1}"

# -------------------------
# Multi-IF params
# -------------------------
# Output log directory for Multi-IF (logs/task_name/model_name).
MULTI_IF_LOG_DIR="${MULTI_IF_LOG_DIR:-${PROJECT_ROOT}/logs/multi_if/${LOG_MODEL_NAME}}"
# Verbose logging for Multi-IF samples.
MULTI_IF_VERBOSE="${MULTI_IF_VERBOSE:-false}"
# Flush interval for partial outputs.
MULTI_IF_FLUSH_EVERY="${MULTI_IF_FLUSH_EVERY:-1}"

CUDA_ARGS=(
    "--cuda_ids=${CUDA_IDS}"
)

log() {
    if [ "${LOG_MAIN_RANK}" -eq 1 ]; then
        echo "$@"
    fi
}

log_header() {
    if [ "${LOG_MAIN_RANK}" -eq 1 ]; then
        echo "----------------------------------------------------"
    fi
}

ensure_dir() {
    local dir="$1"
    if [ ! -d "${dir}" ]; then
        mkdir -p "${dir}"
    fi
}

run_task() {
    local task="$1"
    local mode="$2"
    shift 2
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_eval.py" --task "${task}" --mode "${mode}" -- "$@"
}

case "${TASK}" in
    gsm8k_interference)
        # -------------------------
        # GSM8K Interference params
        # -------------------------
        log_header
        log "Task      : gsm8k_interference"
        log "Mode      : ${MODE}"
        log "CUDA IDS  : ${CUDA_IDS}"
        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            log "Base Model: ${BASE_MODEL_PATH}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            log "Beacon    : ${BEACON_MODEL_PATH}"
            log "Num Sinks : ${NUM_SINKS}"
        fi
        log "Log Dir   : ${GSM8K_LOG_DIR}"
        if [ -n "${GSM8K_PATH}" ]; then
            log "GSM8K Data: ${GSM8K_PATH}"
        else
            log "GSM8K Data: datasets"
        fi
        log "UltraChat : ${GSM8K_ULTRACHAT_PATH}"
        log_header
        ensure_dir "${GSM8K_LOG_DIR}"

        COMMON_ARGS=(
            "--ultrachat_path=${GSM8K_ULTRACHAT_PATH}"
            "--seed=${GSM8K_SEED}"
            "--history_max_turns=${GSM8K_HISTORY_MAX_TURNS}"
            "--max_input_tokens=${GSM8K_MAX_INPUT_TOKENS}"
            "--max_new_tokens=${GSM8K_MAX_NEW_TOKENS}"
            "--temperature=${GSM8K_TEMPERATURE}"
            "--top_p=${GSM8K_TOP_P}"
            "--log_dir=${GSM8K_LOG_DIR}"
        )

        if [ -n "${GSM8K_PATH}" ]; then
            COMMON_ARGS+=("--gsm8k_path=${GSM8K_PATH}")
        fi
        if [ -n "${GSM8K_MAX_SAMPLES}" ]; then
            COMMON_ARGS+=("--max_samples=${GSM8K_MAX_SAMPLES}")
        fi
        if [ -n "${GSM8K_RUN_TAG}" ]; then
            COMMON_ARGS+=("--run_tag=${GSM8K_RUN_TAG}")
        fi
        if [ "${GSM8K_ENABLE_THINKING}" = "true" ]; then
            COMMON_ARGS+=("--enable_thinking=true")
        fi

        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            run_task "gsm8k_interference" "base" \
                "--model_path=${BASE_MODEL_PATH}" \
                "${CUDA_ARGS[@]}" \
                "${COMMON_ARGS[@]}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            run_task "gsm8k_interference" "beacon" \
                "--model_path=${BEACON_MODEL_PATH}" \
                "--num_sinks=${NUM_SINKS}" \
                "${CUDA_ARGS[@]}" \
                "${COMMON_ARGS[@]}"
        fi
        ;;

    mtbench_101)
        # -------------------------
        # MTBench_101 params
        # -------------------------
        log_header
        log "Task      : mtbench_101"
        log "Mode      : ${MODE}"
        log "CUDA IDS  : ${CUDA_IDS}"
        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            log "Base Model: ${BASE_MODEL_PATH}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            log "Beacon    : ${BEACON_MODEL_PATH}"
            log "Num Sinks : ${NUM_SINKS}"
        fi
        log "Log Dir   : ${MTBENCH_LOG_DIR}"
        log "Data Path : ${MTBENCH_DATA_PATH}"
        if [ -n "${MTBENCH_DATA_NAME}" ]; then
            log "Data Name : ${MTBENCH_DATA_NAME}"
        fi
        log "Config    : ${MTBENCH_CONFIG_PATH}"
        log_header
        ensure_dir "${MTBENCH_LOG_DIR}"

        NAME_ARG=()
        if [ -n "${MTBENCH_DATA_NAME}" ]; then
            NAME_ARG+=("--name=${MTBENCH_DATA_NAME}")
        fi
        TOP_P_ARG=()
        if [ -n "${MTBENCH_TOP_P}" ]; then
            TOP_P_ARG+=("--top_p=${MTBENCH_TOP_P}")
        fi

        COMMON_ARGS=(
            "--data_path=${MTBENCH_DATA_PATH}"
            "${NAME_ARG[@]}"
            "--config_path=${MTBENCH_CONFIG_PATH}"
            "${CUDA_ARGS[@]}"
            "--log_dir=${MTBENCH_LOG_DIR}"
            "--max_new_tokens=${MTBENCH_MAX_NEW_TOKENS}"
            "--temperature=${MTBENCH_TEMPERATURE}"
            "--do_sample=${MTBENCH_DO_SAMPLE}"
            "${TOP_P_ARG[@]}"
            "--flush_every=${MTBENCH_FLUSH_EVERY}"
        )

        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            run_task "mtbench_101" "base" \
                "--model_path=${BASE_MODEL_PATH}" \
                "${COMMON_ARGS[@]}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            run_task "mtbench_101" "beacon" \
                "--model_path=${BEACON_MODEL_PATH}" \
                "--num_sinks=${NUM_SINKS}" \
                "${COMMON_ARGS[@]}"
        fi
        ;;

    multi_if)
        # -------------------------
        # Multi-IF params
        # -------------------------
        log_header
        log "Task      : multi_if"
        log "Mode      : ${MODE}"
        log "CUDA IDS  : ${CUDA_IDS}"
        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            log "Base Model: ${BASE_MODEL_PATH}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            log "Beacon    : ${BEACON_MODEL_PATH}"
            log "Num Sinks : ${NUM_SINKS}"
        fi
        log "Log Dir   : ${MULTI_IF_LOG_DIR}"
        log "Verbose   : ${MULTI_IF_VERBOSE}"
        log_header
        ensure_dir "${MULTI_IF_LOG_DIR}"

        COMMON_ARGS=(
            "${CUDA_ARGS[@]}"
            "--log_dir=${MULTI_IF_LOG_DIR}"
            "--flush_every=${MULTI_IF_FLUSH_EVERY}"
        )
        if [ "${MULTI_IF_VERBOSE}" = "true" ]; then
            COMMON_ARGS+=("--verbose=true")
        fi

        if [ "${MODE}" = "base" ] || [ "${MODE}" = "both" ]; then
            run_task "multi_if" "base" \
                "--model_path=${BASE_MODEL_PATH}" \
                "${COMMON_ARGS[@]}"
        fi
        if [ "${MODE}" = "beacon" ] || [ "${MODE}" = "both" ]; then
            run_task "multi_if" "beacon" \
                "--model_path=${BEACON_MODEL_PATH}" \
                "--num_sinks=${NUM_SINKS}" \
                "${COMMON_ARGS[@]}"
        fi
        ;;

    *)
        echo "Unknown TASK: ${TASK}"
        exit 1
        ;;
esac
