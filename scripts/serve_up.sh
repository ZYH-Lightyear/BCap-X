#!/usr/bin/env bash
# Bring up RoboMEx/CapX prerequisite services in a detachable tmux session.
#
#   window 0 "proxy" : OpenAI-compatible LLM proxy on :8110 (the port every client expects)
#   window 1 "gpu"   : sam3 / graspnet / pyroki via capx.serving.launch_servers
#
# Usage:
#   scripts/serve_up.sh [CONFIG_YAML]
#   scripts/serve_up.sh configs/services/vapi.env
#
# Env overrides:
#   SESSION       tmux session name        (default: robomex)
#   LLM_PORT      LLM proxy port           (default: 8110)
#   LLM_BASE_URL  upstream OpenAI-compatible base URL
#   LLM_API_KEY   upstream API key, preferred over KEY_FILE when set
#   LLM_API_KEY_ENV name of an env var containing the upstream API key (e.g. V_API_KEY)
#   KEY_FILE      fallback key file        (default: .openrouterkey)
#   LOG_DIR       per-server log directory (default: ./logs/servers)
set -euo pipefail

ARG="${1:-}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(readlink -f "$0")")")"
REPO_ROOT="$PWD"

# Load local secrets/config if present. .env is gitignored; keep API keys there.
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    set -a
    source .env
    set +a
fi

# First argument can either be the legacy LIBERO YAML or a shell-style service config.
# A service config may set CONFIG=env_configs/...yaml and LLM_* variables.
if [ -n "$ARG" ]; then
    if [ ! -f "$ARG" ]; then
        echo "ERROR: config '$ARG' not found"
        exit 1
    fi
    case "$ARG" in
        *.yaml|*.yml)
            CONFIG="$ARG"
            ;;
        *)
            # shellcheck disable=SC1090
            set -a
            source "$ARG"
            set +a
            CONFIG="${CONFIG:-env_configs/libero/franka_libero_spatial_0.yaml}"
            ;;
    esac
else
    CONFIG="${CONFIG:-env_configs/libero/franka_libero_spatial_0.yaml}"
fi

SESSION="${SESSION:-robomex}"
LLM_PORT="${LLM_PORT:-8110}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1/}}"
LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-low}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-600}"
KEY_FILE="${KEY_FILE:-${OPENROUTER_KEY_FILE:-.openrouterkey}}"
LOG_DIR="${LOG_DIR:-./logs/servers}"
# Virtualenv to run inside. .venv-libero is the complete env (torch + sam3 +
# graspnet + pyroki); the bare project .venv is incomplete.
VENV="${VENV:-.venv-libero}"
RUN="uv run --no-sync --active"

# Restore the persistent uv toolchain (the container root FS is wiped on reboot).
# shellcheck source=scripts/env.sh
source scripts/env.sh

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux not installed"; exit 1; }
[ -d "$VENV" ] || { echo "ERROR: venv '$VENV' not found (see docs/libero-tasks.md to create it)"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: config '$CONFIG' not found"; exit 1; }

if [ -n "${LLM_API_KEY_ENV:-}" ]; then
    LLM_API_KEY="${!LLM_API_KEY_ENV:-${LLM_API_KEY:-}}"
fi

if [ -z "${LLM_API_KEY:-}" ] && [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: neither LLM_API_KEY nor key file '$KEY_FILE' is available"
    echo "       Set LLM_API_KEY, or LLM_API_KEY_ENV=YOUR_ENV_VAR with that env var exported,"
    echo "       or provide KEY_FILE=.openrouterkey."
    exit 1
fi
export LLM_API_KEY

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already running -> tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Each tmux window is a fresh login-less shell with none of our env, so cd to
# the repo, re-source the persistent uv toolchain, and activate the LIBERO venv.
PREP="cd $REPO_ROOT && source scripts/env.sh && source $VENV/bin/activate"

# Bake the resolved key directly into the tmux command (safely quoted). The
# tmux pane is a fresh shell that does NOT inherit LLM_API_KEY from us, so a
# literal "$LLM_API_KEY" would expand to empty there and the upstream rejects
# the request with 401 "no token provided".
if [ -n "${LLM_API_KEY:-}" ]; then
    LLM_AUTH_ARGS="--api-key $(printf '%q' "$LLM_API_KEY")"
else
    LLM_AUTH_ARGS="--key-file $KEY_FILE"
fi

tmux new-session -d -s "$SESSION" -n proxy
tmux send-keys -t "$SESSION:proxy" \
    "$PREP && $RUN python -m capx.serving.openrouter_server $LLM_AUTH_ARGS --base-url $LLM_BASE_URL --host 0.0.0.0 --port $LLM_PORT --reasoning-effort $LLM_REASONING_EFFORT --timeout-s $LLM_TIMEOUT_S" C-m

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
echo "  LLM proxy: :$LLM_PORT -> $LLM_BASE_URL"
echo "  LIBERO config: $CONFIG"
echo "  attach: tmux attach -t $SESSION   (Ctrl-b d to detach, Ctrl-b n to switch window)"
echo "  stop:   scripts/serve_down.sh"
