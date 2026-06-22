#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/xuyingjie/BCap-X"
DATA_DIR="${ROOT}/data"
CKPT_DIR="${ROOT}/checkpoint"
B1K_DIR="${ROOT}/capx/third_party/b1k"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${CKPT_DIR}/tmp"

# Keep BEHAVIOR / OmniGibson datasets in data/ and HuggingFace checkpoints in checkpoint/.
export OMNIGIBSON_DATA_PATH="${DATA_DIR}"
export HF_HOME="${CKPT_DIR}"
export TMPDIR="${CKPT_DIR}/tmp"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${ROOT}"
git submodule update --init --recursive capx/third_party/b1k capx/third_party/curobo capx/third_party/sam3 capx/third_party/contact_graspnet_pytorch

cd "${B1K_DIR}"
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate

# ./uv_install.sh --dataset --accept-dataset-tos

# Post-install cuRobo header fix from README.
cp /mnt/data/xuyingjie/BCap-X/capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
  "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/curobo/curobolib/cpp/"

echo ""
echo "Done. Data dir: ${DATA_DIR}"
echo "Done. Checkpoint/cache dir: ${CKPT_DIR}"
