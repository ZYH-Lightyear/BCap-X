#!/usr/bin/env bash
# Bring up RoboMEx/CapX prerequisite services in a detachable tmux session.
#
#   window 0 "proxy" : OpenRouter LLM proxy on :8110 (the port every client expects)
#   window 1 "gpu"   : sam3 / graspnet / pyroki via capx.serving.launch_servers
#
# Usage:
#   scripts/serve_up.sh [CONFIG_YAML]
#
# Env overrides:
#   SESSION   tmux session name        (default: robomex)
#   LLM_PORT  LLM proxy port           (default: 8110)
#   KEY_FILE  OpenRouter key file      (default: .openrouterkey)
#   LOG_DIR   per-server log directory (default: ./logs/servers)
set -euo pipefail

SESSION="${SESSION:-robomex}"
LLM_PORT="${LLM_PORT:-8110}"
KEY_FILE="${KEY_FILE:-.openrouterkey}"
LOG_DIR="${LOG_DIR:-./logs/servers}"
CONFIG="${1:-env_configs/libero/franka_libero_spatial_0.yaml}"
# Virtualenv to run inside. .venv-libero is the complete env (torch + sam3 +
# graspnet + pyroki); the bare project .venv is incomplete.
VENV="${VENV:-.venv-libero}"
RUN="uv run --no-sync --active"

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(readlink -f "$0")")")"
REPO_ROOT="$PWD"

# Restore the persistent uv toolchain (the container root FS is wiped on reboot).
# shellcheck source=scripts/env.sh
source scripts/env.sh

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not installed"; exit 1; }
[ -d "$VENV" ] || { echo "ERROR: venv '$VENV' not found (see docs/libero-tasks.md to create it)"; exit 1; }
[ -f "$KEY_FILE" ] || { echo "ERROR: key file '$KEY_FILE' not found"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: config '$CONFIG' not found"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already running -> tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Each tmux window is a fresh login-less shell with none of our env, so cd to
# the repo, re-source the persistent uv toolchain, and activate the LIBERO venv.
PREP="cd $REPO_ROOT && source scripts/env.sh && source $VENV/bin/activate"

tmux new-session -d -s "$SESSION" -n proxy
tmux send-keys -t "$SESSION:proxy" \
    "$PREP && $RUN python -m capx.serving.openrouter_server --key-file $KEY_FILE --host 0.0.0.0 --port $LLM_PORT" C-m

tmux new-window -t "$SESSION" -n gpu
tmux send-keys -t "$SESSION:gpu" \
    "$PREP && $RUN python capx/serving/launch_servers.py --config-path $CONFIG --log-dir $LOG_DIR" C-m

# Wait for the LLM proxy to accept connections (the GPU launcher waits on its own ports).
echo -n "waiting for LLM proxy on :$LLM_PORT "
for _ in $(seq 1 60); do
    if (echo > "/dev/tcp/127.0.0.1/$LLM_PORT") 2>/dev/null; then
        echo "ready"
        break
    fi
    echo -n "."
    sleep 2
done

echo
echo "services starting in tmux session '$SESSION'"
echo "  attach: tmux attach -t $SESSION   (Ctrl-b d to detach, Ctrl-b n to switch window)"
echo "  stop:   scripts/serve_down.sh"
