#!/usr/bin/env bash
# Start the four local services needed by LIBERO / CaP-Agent0 tests.
#
# Services:
#   8110: OpenRouter proxy
#   8114: SAM3
#   8115: Contact-GraspNet
#   8116: PyRoKi
#
# Usage:
#   bash scripts/start_libero_services.sh
#
# Optional:
#   OPENROUTER_KEY_FILE=.openrouterkey bash scripts/start_libero_services.sh
#   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/ bash scripts/start_libero_services.sh
#   FORCE_RESTART=1 bash scripts/start_libero_services.sh
#   SAM3_CUDA_DEVICES=3 GRASPNET_CUDA_DEVICES=3 bash scripts/start_libero_services.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

OPENROUTER_KEY_FILE="${OPENROUTER_KEY_FILE:-.openrouterkey}"
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1/}"
FORCE_RESTART="${FORCE_RESTART:-0}"

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

AUTO_CUDA_DEVICE="$(pick_least_used_gpu)"
SAM3_CUDA_DEVICES="${SAM3_CUDA_DEVICES:-$AUTO_CUDA_DEVICE}"
GRASPNET_CUDA_DEVICES="${GRASPNET_CUDA_DEVICES:-$AUTO_CUDA_DEVICE}"

if [ ! -d ".venv-libero" ]; then
    echo "ERROR: .venv-libero not found."
    echo "Create/install the LIBERO env first, then rerun this script."
    exit 1
fi

source .venv-libero/bin/activate

port_up() {
    local port="$1"
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        return 0
    fi
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
        return 0
    fi
    if curl -sf --connect-timeout 2 "http://127.0.0.1:${port}/" >/dev/null 2>&1; then
        return 0
    fi
    if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

kill_port_if_requested() {
    local port="$1"
    local name="$2"

    if [ "$FORCE_RESTART" != "1" ]; then
        return 0
    fi

    if ! port_up "$port"; then
        return 0
    fi

    echo "  ${name} (${port}): FORCE_RESTART=1, stopping existing process..."
    if command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill $pids 2>/dev/null || true
        fi
    elif command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    else
        echo "    WARNING: neither lsof nor fuser is available; cannot stop port ${port} automatically."
    fi

    sleep 3
}

start_if_down() {
    local port="$1"
    local name="$2"
    local logfile="$3"
    shift 3

    kill_port_if_requested "$port" "$name"

    if port_up "$port"; then
        echo "  ${name} (${port}): already UP"
        return 0
    fi

    echo "  ${name} (${port}): starting..."
    nohup "$@" > "$logfile" 2>&1 &
    echo "    pid=$! log=$logfile"
}

echo "========================================================================"
echo "Starting LIBERO services"
echo "========================================================================"
echo "Auto-selected CUDA device: $AUTO_CUDA_DEVICE"
echo "SAM3 CUDA_VISIBLE_DEVICES: $SAM3_CUDA_DEVICES"
echo "GraspNet CUDA_VISIBLE_DEVICES: $GRASPNET_CUDA_DEVICES"
echo "FORCE_RESTART: $FORCE_RESTART"
echo "========================================================================"

if [ ! -f "$OPENROUTER_KEY_FILE" ]; then
    echo "WARNING: $OPENROUTER_KEY_FILE not found."
    echo "OpenRouter proxy will fail unless you set OPENROUTER_KEY_FILE or create .openrouterkey."
fi

start_if_down 8110 "OpenRouter proxy" "logs/openrouter_8110.log" \
    uv run --no-sync --active python -m capx.serving.openrouter_server \
        --key-file "$OPENROUTER_KEY_FILE" \
        --host 127.0.0.1 \
        --port 8110 \
        --base-url "$OPENROUTER_BASE_URL"

start_if_down 8114 "SAM3" "logs/sam3_8114.log" \
    env CUDA_VISIBLE_DEVICES="$SAM3_CUDA_DEVICES" \
    uv run --no-sync --active python -m capx.serving.launch_sam3_server \
        --device cuda \
        --port 8114 \
        --host 127.0.0.1

start_if_down 8115 "Contact-GraspNet" "logs/graspnet_8115.log" \
    env CUDA_VISIBLE_DEVICES="$GRASPNET_CUDA_DEVICES" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server \
        --port 8115 \
        --host 127.0.0.1

start_if_down 8116 "PyRoKi" "logs/pyroki_8116.log" \
    uv run --no-sync --active python -m capx.serving.launch_pyroki_server \
        --port 8116 \
        --host 127.0.0.1 \
        --robot panda_description \
        --target-link panda_hand

echo ""
echo "Waiting for services to become reachable..."

deadline=$((SECONDS + 180))
while true; do
    all_up=true
    for item in "8110:OpenRouter proxy" "8114:SAM3" "8115:Contact-GraspNet" "8116:PyRoKi"; do
        port="${item%%:*}"
        name="${item#*:}"
        if port_up "$port"; then
            status="UP"
        else
            status="DOWN"
            all_up=false
        fi
        printf "  %-18s (%s): %s\n" "$name" "$port" "$status"
    done

    if [ "$all_up" = true ]; then
        echo ""
        echo "All LIBERO services are UP."
        echo "Now run:"
        echo "  bash scripts/test_libero_cap_agent0.sh"
        exit 0
    fi

    if [ "$SECONDS" -ge "$deadline" ]; then
        echo ""
        echo "ERROR: Some services did not become ready within 180s."
        echo "Check logs:"
        echo "  logs/openrouter_8110.log"
        echo "  logs/sam3_8114.log"
        echo "  logs/graspnet_8115.log"
        echo "  logs/pyroki_8116.log"
        exit 1
    fi

    echo "  ...retrying in 5s"
    sleep 5
done
