#!/usr/bin/env bash
# Bring up RoboMEx/CapX prerequisite services in a detachable tmux session.
#
#   window 0 "proxy" : OpenAI-compatible LLM proxy on :8110 (the port every client expects)
#   window 1 "gpu"   : sam3 / graspnet / pyroki via capx.serving.launch_servers
#   window 2 "webui" : Cap-X chat Web UI on :8200
#   window 3 "trace" : RoboMEx Swarm Observatory on :8300
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
#   WEBUI_PORT    Cap-X web UI port        (default: 8200)
#   TRACE_PORT    RoboMEx Trace UI port    (default: 8300)
#   TRACE_ROOT    run scan root            (default: outputs/robomex_libero_live)
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
# Multi-upstream routing: when set, the proxy serves several backends at once,
# routed by model prefix (openrouter/... vs vapi/...). Keys are resolved by the
# proxy itself from .env / key files, so no key needs baking into the command.
LLM_ROUTES_FILE="${LLM_ROUTES_FILE:-configs/services/llm_routes.json}"
KEY_FILE="${KEY_FILE:-${OPENROUTER_KEY_FILE:-.openrouterkey}}"
LOG_DIR="${LOG_DIR:-./logs/servers}"
WEBUI_PORT="${WEBUI_PORT:-8200}"
TRACE_PORT="${TRACE_PORT:-8300}"
TRACE_ROOT="${TRACE_ROOT:-outputs/robomex_libero_live}"
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

# Routes-file mode resolves keys inside the proxy (from .env / key files), so the
# single-key requirement only applies to legacy single-upstream mode.
if [ -z "$LLM_ROUTES_FILE" ]; then
    if [ -z "${LLM_API_KEY:-}" ] && [ ! -f "$KEY_FILE" ]; then
        echo "ERROR: neither LLM_API_KEY nor key file '$KEY_FILE' is available"
        echo "       Set LLM_API_KEY, or LLM_API_KEY_ENV=YOUR_ENV_VAR with that env var exported,"
        echo "       or provide KEY_FILE=.openrouterkey."
        exit 1
    fi
    export LLM_API_KEY
elif [ ! -f "$LLM_ROUTES_FILE" ]; then
    echo "ERROR: LLM_ROUTES_FILE='$LLM_ROUTES_FILE' not found"
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already running -> tmux attach -t $SESSION"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Each tmux window is a fresh login-less shell with none of our env, so cd to
# the repo, re-source the persistent uv toolchain, load repo-local secrets, and
# activate the LIBERO venv. ``set -a`` exports keys from .env without embedding
# their values in the tmux command line.
PREP="cd $REPO_ROOT && source scripts/env.sh && if [ -f .env ]; then set -a && source .env && set +a; fi && source $VENV/bin/activate"

# Build the proxy launch args. In routes-file mode the proxy reads its own keys
# (from .env / key files) and routes by model prefix; the single-upstream args
# (key/base-url/effort) are then irrelevant. In legacy mode, bake the resolved
# key directly into the command (safely quoted) -- the tmux pane is a fresh shell
# that does NOT inherit LLM_API_KEY, so a literal "$LLM_API_KEY" would expand to
# empty there and the upstream rejects the request with 401 "no token provided".
if [ -n "$LLM_ROUTES_FILE" ]; then
    LLM_PROXY_ARGS="--routes-file $(printf '%q' "$LLM_ROUTES_FILE")"
else
    if [ -n "${LLM_API_KEY:-}" ]; then
        LLM_AUTH_ARGS="--api-key $(printf '%q' "$LLM_API_KEY")"
    else
        LLM_AUTH_ARGS="--key-file $KEY_FILE"
    fi
    LLM_PROXY_ARGS="$LLM_AUTH_ARGS --base-url $LLM_BASE_URL --reasoning-effort $LLM_REASONING_EFFORT"
fi

tmux new-session -d -s "$SESSION" -n proxy
tmux send-keys -t "$SESSION:proxy" \
    "$PREP && $RUN python -m capx.serving.openrouter_server $LLM_PROXY_ARGS --dotenv-path .env --host 0.0.0.0 --port $LLM_PORT --timeout-s $LLM_TIMEOUT_S" C-m

tmux new-window -t "$SESSION" -n gpu
tmux send-keys -t "$SESSION:gpu" \
    "$PREP && $RUN python capx/serving/launch_servers.py --config-path $CONFIG --log-dir $LOG_DIR" C-m

tmux new-window -t "$SESSION" -n webui
tmux send-keys -t "$SESSION:webui" \
    "$PREP && python -m capx.web.server --host 0.0.0.0 --port $WEBUI_PORT" C-m

# Swarm Observatory: build SPA if missing, then serve read-only API + static UI.
# Access from your laptop via http://<server-ip>:$TRACE_PORT (not localhost).
TRACE_BOOT='if [ ! -f robomex-ui/dist/index.html ]; then'
TRACE_BOOT+=' if command -v npm >/dev/null 2>&1; then'
TRACE_BOOT+=' (cd robomex-ui && npm install --no-fund --no-audit && npm run build);'
TRACE_BOOT+=' else echo "WARN: robomex-ui/dist missing and npm not found"; fi; fi'
tmux new-window -t "$SESSION" -n trace
tmux send-keys -t "$SESSION:trace" \
    "$PREP && $TRACE_BOOT && $RUN robomex ui --host 0.0.0.0 --port $TRACE_PORT --root $TRACE_ROOT" C-m

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

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-<server-ip>}"

echo
echo "services starting in tmux session '$SESSION'"
echo "  LLM proxy:   :$LLM_PORT -> $LLM_BASE_URL"
echo "  Cap-X WebUI: :$WEBUI_PORT -> http://$HOST_IP:$WEBUI_PORT"
echo "  Observatory: :$TRACE_PORT -> http://$HOST_IP:$TRACE_PORT"
echo "  LIBERO config: $CONFIG"
echo "  attach: tmux attach -t $SESSION   (Ctrl-b d to detach, Ctrl-b n to switch window)"
echo "  stop:   scripts/serve_down.sh"
