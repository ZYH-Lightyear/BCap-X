#!/usr/bin/env bash
# Stop the RoboMEx/CapX prerequisite services started by serve_up.sh.
#
# Usage: scripts/serve_down.sh
# Env:   SESSION  tmux session name (default: robomex)
set -euo pipefail

SESSION="${SESSION:-robomex}"
PORTS="${PORTS:-8110 8114 8115 8116}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "stopped tmux session '$SESSION'"
else
    echo "no tmux session '$SESSION' running"
fi

for port in $PORTS; do
    pids="$(
        ss -ltnp "sport = :$port" 2>/dev/null \
            | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
            | sort -u
    )"
    if [ -z "$pids" ]; then
        echo "port $port: free"
        continue
    fi

    echo "port $port: stopping pid(s) $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
done

sleep 2

for port in $PORTS; do
    pids="$(
        ss -ltnp "sport = :$port" 2>/dev/null \
            | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
            | sort -u
    )"
    if [ -n "$pids" ]; then
        echo "port $port: force killing pid(s) $pids"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
done
