#!/usr/bin/env bash
# Persistent uv environment for this Aliyun PAI-DSW container.
#
# WHY THIS EXISTS
#   The container root filesystem ("/", incl. /usr/local/bin and /root) is an
#   ephemeral overlay that is wiped on every restart. Only /mnt/data (CPFS) is
#   persistent. A stock `uv` install therefore disappears on reboot:
#     - the uv binary (/usr/local/bin/uv) is gone   -> "uv: command not found"
#     - uv-managed Pythons (/root/.local/share/uv)  -> ".venv" interpreter dies
#     - the uv cache (/root/.cache/uv)              -> rebuilds are slow
#
#   This script pins uv's binary, managed Pythons and cache onto /mnt/data so
#   everything survives a restart. Source it once per shell after a reboot:
#
#       source scripts/env.sh
#
#   It is also sourced automatically by scripts/serve_up.sh and friends.
#
# Override UV_ROOT to relocate the persistent store.

# Resolve repo-independent persistent location (must live under a CPFS mount).
export UV_ROOT="${UV_ROOT:-/mnt/data/zyh/.uv}"

# Keep uv-managed Pythons, cache and tools on persistent storage.
export UV_PYTHON_INSTALL_DIR="$UV_ROOT/python"
export UV_CACHE_DIR="$UV_ROOT/cache"
export UV_TOOL_DIR="$UV_ROOT/tools"
export UV_TOOL_BIN_DIR="$UV_ROOT/bin"
# Prefer uv-managed (persistent) interpreters over ephemeral system ones.
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"

mkdir -p "$UV_ROOT/bin" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_TOOL_DIR"

# Put the persistent uv binary first on PATH.
case ":$PATH:" in
    *":$UV_ROOT/bin:"*) ;;
    *) export PATH="$UV_ROOT/bin:$PATH" ;;
esac

# Self-heal: if the persistent uv binary vanished, recover it.
if [ ! -x "$UV_ROOT/bin/uv" ]; then
    if command -v uv >/dev/null 2>&1 && [ "$(command -v uv)" != "$UV_ROOT/bin/uv" ]; then
        echo "[env] copying uv -> $UV_ROOT/bin" >&2
        cp -f "$(command -v uv)" "$UV_ROOT/bin/uv" 2>/dev/null || true
        cp -f "$(command -v uvx)" "$UV_ROOT/bin/uvx" 2>/dev/null || true
    else
        echo "[env] uv missing, downloading into $UV_ROOT/bin ..." >&2
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$UV_ROOT/bin" INSTALLER_NO_MODIFY_PATH=1 sh
    fi
fi

if command -v uv >/dev/null 2>&1; then
    echo "[env] uv ready: $(command -v uv) ($(uv --version 2>/dev/null))" >&2
else
    echo "[env] WARNING: uv still not on PATH" >&2
fi

# LIBERO also depends on a couple of root-filesystem paths. Recreate them here
# so a fresh DSW container can run after a single `source scripts/env.sh`.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." 2>/dev/null && pwd)"

if [ -n "$_REPO_ROOT" ] && [ -d "$_REPO_ROOT/capx/third_party/LIBERO-PRO/libero/libero" ]; then
    _LIBERO_ROOT="$_REPO_ROOT/capx/third_party/LIBERO-PRO/libero/libero"
    mkdir -p "$HOME/.libero"
    cat > "$HOME/.libero/config.yaml" << EOF
assets: $_LIBERO_ROOT/assets
bddl_files: $_LIBERO_ROOT/bddl_files
benchmark_root: $_LIBERO_ROOT
datasets: $_LIBERO_ROOT/../datasets
init_states: $_LIBERO_ROOT/init_files
EOF
    echo "[env] LIBERO config ready: $HOME/.libero/config.yaml" >&2
    unset _LIBERO_ROOT
fi

# MuJoCo/robosuite EGL offscreen rendering needs this NVIDIA vendor ICD. On this
# server it can disappear with the ephemeral root filesystem even though the
# NVIDIA EGL shared library is still present.
if [ -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ] && [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    if mkdir -p /usr/share/glvnd/egl_vendor.d 2>/dev/null; then
        cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
        echo "[env] EGL vendor ICD ready: /usr/share/glvnd/egl_vendor.d/10_nvidia.json" >&2
    else
        echo "[env] WARNING: cannot create /usr/share/glvnd/egl_vendor.d; EGL may fail" >&2
    fi
fi

unset _SCRIPT_DIR _REPO_ROOT
