#!/usr/bin/env bash
# Quick test for LIBERO CaP-Agent0.
#
# This script assumes the required services are already running:
#   8110: OpenRouter proxy / OpenAI-compatible LLM proxy
#   8114: SAM3
#   8115: Contact-GraspNet
#   8116: PyRoKi
#
# Usage:
#   bash scripts/test_libero_cap_agent0.sh
#
# Useful overrides:
#   TRIALS=1 WORKERS=1 bash scripts/test_libero_cap_agent0.sh
#   BACKEND=molmo bash scripts/test_libero_cap_agent0.sh
#   MODEL="openrouter/qwen/qwen3.6-plus" bash scripts/test_libero_cap_agent0.sh
#   OUTPUT_DIR="./outputs/franka_libero_cap_agent0_debug" bash scripts/test_libero_cap_agent0.sh
#   CUDA_DEVICES=auto bash scripts/test_libero_cap_agent0.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p logs

pick_least_used_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return 0
    fi

    local gpu
    gpu="$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk -F',' '
                NF >= 2 {
                    gsub(/ /, "", $1);
                    gsub(/ /, "", $2);
                    if (best == "" || $2 < best_mem) {
                        best = $1;
                        best_mem = $2;
                    }
                }
                END {
                    if (best == "") print "0";
                    else print best;
                }
            '
    )"
    echo "${gpu:-0}"
}

CONFIG_PATH="${CONFIG_PATH:-env_configs/libero/franka_libero_cap_agent0.yaml}"
MODEL="${MODEL:-openrouter/qwen/qwen3.6-plus}"
VDM_MODEL="${VDM_MODEL:-$MODEL}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
VDM_SERVER_URL="${VDM_SERVER_URL:-$SERVER_URL}"
BACKEND="${BACKEND:-qwen}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"
TRIALS="${TRIALS:-3}"
WORKERS="${WORKERS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
MAX_RETRIES="${MAX_RETRIES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/franka_libero_cap_agent0_quick_test}"
CUDA_DEVICES="${CUDA_DEVICES:-auto}"
RECORD_VIDEO="${RECORD_VIDEO:-True}"
USE_VISUAL_FEEDBACK="${USE_VISUAL_FEEDBACK:-False}"
USE_IMG_DIFFERENCING="${USE_IMG_DIFFERENCING:-True}"
USE_PARALLEL_ENSEMBLE="${USE_PARALLEL_ENSEMBLE:-True}"
USE_MULTIMODEL="${USE_MULTIMODEL:-True}"

normalize_bool() {
    case "$1" in
        true|TRUE|True|1|yes|YES|Yes) echo "True" ;;
        false|FALSE|False|0|no|NO|No) echo "False" ;;
        none|NONE|None|null|NULL|Null) echo "None" ;;
        *)
            echo "ERROR: invalid boolean value '$1'. Use True/False/None." >&2
            return 1
            ;;
    esac
}

RECORD_VIDEO="$(normalize_bool "$RECORD_VIDEO")"
USE_VISUAL_FEEDBACK="$(normalize_bool "$USE_VISUAL_FEEDBACK")"
USE_IMG_DIFFERENCING="$(normalize_bool "$USE_IMG_DIFFERENCING")"
USE_PARALLEL_ENSEMBLE="$(normalize_bool "$USE_PARALLEL_ENSEMBLE")"
USE_MULTIMODEL="$(normalize_bool "$USE_MULTIMODEL")"

if [ "$CUDA_DEVICES" = "auto" ] || [ "$CUDA_DEVICES" = "AUTO" ]; then
    CUDA_DEVICES="$(pick_least_used_gpu)"
fi

echo "========================================================================"
echo "LIBERO CaP-Agent0 Quick Test"
echo "========================================================================"
echo "Config:                 $CONFIG_PATH"
echo "Model:                  $MODEL"
echo "Visual differencing:    $VDM_MODEL"
echo "Point backend:          $BACKEND"
echo "Max tokens:             $MAX_TOKENS"
echo "Reasoning effort:       $REASONING_EFFORT"
echo "Trials / workers:       $TRIALS / $WORKERS"
echo "Trial timeout / retries: ${TIMEOUT_SECONDS}s / $MAX_RETRIES"
echo "Record video:           $RECORD_VIDEO"
echo "CUDA devices:           $CUDA_DEVICES"
echo "Output dir:             $OUTPUT_DIR"
echo "========================================================================"
echo ""

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -d ".venv-libero" ]; then
    echo "ERROR: .venv-libero not found. Create/activate the LIBERO environment first."
    exit 1
fi

check_http_or_tcp() {
    local port="$1"
    local label="$2"
    local required="${3:-true}"

    local ok=false
    if curl -sf --connect-timeout 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        ok=true
    elif curl -sf --connect-timeout 3 "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
        ok=true
    elif curl -sf --connect-timeout 3 "http://127.0.0.1:${port}/" >/dev/null 2>&1; then
        ok=true
    elif timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
        ok=true
    fi

    if [ "$ok" = true ]; then
        echo "  ${label} (${port}): UP"
        return 0
    fi

    echo "  ${label} (${port}): DOWN"
    if [ "$required" = true ]; then
        return 1
    fi
    return 0
}

echo "=== Checking required services ==="
missing=false
check_http_or_tcp 8110 "OpenRouter proxy" || missing=true
check_http_or_tcp 8114 "SAM3" || missing=true
check_http_or_tcp 8115 "Contact-GraspNet" || missing=true
check_http_or_tcp 8116 "PyRoKi" || missing=true

if [ "$missing" = true ]; then
    echo ""
    echo "ERROR: Some required services are down."
    echo ""
    echo "Start them in separate terminals first, for example:"
    echo "  source .venv-libero/bin/activate"
    echo "  uv run --no-sync --active python -m capx.serving.openrouter_server --key-file .openrouterkey --port 8110"
    echo "  uv run --no-sync --active python -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1"
    echo "  uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1"
    echo "  uv run --no-sync --active python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand"
    echo ""
    echo "Tip: run the evaluation only after all four ports are UP."
    exit 1
fi

echo ""
echo "=== Launching CaP-Agent0 evaluation ==="
echo "Logs will also be written to logs/test_libero_cap_agent0.log"
echo ""

source .venv-libero/bin/activate

CAPX_POINT_BACKEND="$BACKEND" \
CAPX_TRIAL_TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
CAPX_TRIAL_MAX_RETRIES="$MAX_RETRIES" \
MUJOCO_GL=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
uv run --no-sync --active capx/envs/launch.py \
    --config-path "$CONFIG_PATH" \
    --model "$MODEL" \
    --server-url "$SERVER_URL" \
    --max-tokens "$MAX_TOKENS" \
    --reasoning-effort "$REASONING_EFFORT" \
    --visual-differencing-model "$VDM_MODEL" \
    --visual-differencing-model-server-url "$VDM_SERVER_URL" \
    --total-trials "$TRIALS" \
    --num-workers "$WORKERS" \
    --record-video "$RECORD_VIDEO" \
    --use-visual-feedback "$USE_VISUAL_FEEDBACK" \
    --use-img-differencing "$USE_IMG_DIFFERENCING" \
    --use-parallel-ensemble "$USE_PARALLEL_ENSEMBLE" \
    --use-multimodel "$USE_MULTIMODEL" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee logs/test_libero_cap_agent0.log

echo ""
echo "=== Done ==="
echo "Output root:"
echo "  $OUTPUT_DIR"
echo ""
echo "Note: launch.py inserts the model name into the output path, so actual results are usually under:"
echo "  outputs/$(echo "$MODEL" | tr '/' '_')/$(basename "$OUTPUT_DIR")"
echo ""
echo "Quick result count:"
find outputs -path "*$(basename "$OUTPUT_DIR")*" -name "trial_*" -type d 2>/dev/null | wc -l | awk '{print "  trial dirs: " $1}'
find outputs -path "*$(basename "$OUTPUT_DIR")*" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l | awk '{print "  successes:  " $1}'
