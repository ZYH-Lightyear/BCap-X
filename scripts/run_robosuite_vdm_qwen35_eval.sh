#!/usr/bin/env bash
# CapX-standard Robosuite multiturn-VDM success-rate eval with qwen3.5-397b-a17b
# (same model/server stack as the cube_stack smoke).
#
# Usage:
#   bash scripts/run_robosuite_vdm_qwen35_eval.sh
#   NUM_WORKERS=4 TOTAL_TRIALS=100 bash scripts/run_robosuite_vdm_qwen35_eval.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

source /Knowin/foundation/bohanzhou/__backup/ENV/miniconda3/etc/profile.d/conda.sh
conda activate sci

export PYTHONPATH="$PWD:$PWD/capx/third_party/pyroki/src:$PWD/capx/third_party/sam3:${PYTHONPATH:-}"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://101.132.143.105:6068}"
export http_proxy="${http_proxy:-http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ALL_PROXY="$http_proxy"

MODEL="${MODEL:-qwen3.5-397b-a17b}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
# Per-task trial count (override with TOTAL_TRIALS=...). Workers kept moderate for DashScope.
TOTAL_TRIALS="${TOTAL_TRIALS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_FILE="${LOG_FILE:-logs/robosuite_vdm_qwen35_eval.log}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/${MODEL}/robosuite_vdm_success_rates.txt}"

CONFIGS=(
  "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm.yaml"
  "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml"
  "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm.yaml"
  "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm.yaml"
  "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm.yaml"
  "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm.yaml"
  "env_configs/two_arm_handover/two_arm_handover_multiturn_vdm.yaml"
)

echo "============================================================" | tee -a "$LOG_FILE"
echo "Robosuite CapX multiturn-VDM eval" | tee -a "$LOG_FILE"
echo "Model: $MODEL" | tee -a "$LOG_FILE"
echo "Server: $SERVER_URL" | tee -a "$LOG_FILE"
echo "VDM: $MODEL @ $SERVER_URL" | tee -a "$LOG_FILE"
echo "Trials/task: $TOTAL_TRIALS  Workers: $NUM_WORKERS" | tee -a "$LOG_FILE"
echo "SAM3: $SAM3_SERVICE_URL" | tee -a "$LOG_FILE"
echo "Configs: ${#CONFIGS[@]}" | tee -a "$LOG_FILE"
echo "Started: $(date -Is)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Preconditions
python - <<'PY'
import socket, sys, urllib.request
for p in (8110, 8115, 8116):
    try:
        s = socket.create_connection(("127.0.0.1", p), 2); s.close()
    except OSError as e:
        print(f"ERROR: port {p} not ready: {e}", file=sys.stderr)
        sys.exit(1)
req = urllib.request.Request("http://127.0.0.1:8110/health")
with urllib.request.urlopen(req, timeout=5) as r:
    print("LLM proxy health:", r.read().decode())
PY

failed=()
idx=0
for cfg in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  stem="$(basename "$cfg" .yaml)"
  echo "" | tee -a "$LOG_FILE"
  echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "[$idx/${#CONFIGS[@]}] $cfg" | tee -a "$LOG_FILE"
  echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

  if python capx/envs/launch.py \
      --config-path "$cfg" \
      --model "$MODEL" \
      --server-url "$SERVER_URL" \
      --visual-differencing-model "$MODEL" \
      --visual-differencing-model-server-url "$SERVER_URL" \
      --total-trials "$TOTAL_TRIALS" \
      --num-workers "$NUM_WORKERS" \
      2>&1 | tee -a "$LOG_FILE"; then
    echo "[OK] $stem" | tee -a "$LOG_FILE"
  else
    echo "[FAIL] $stem (exit $?)" | tee -a "$LOG_FILE"
    failed+=("$stem")
  fi
done

# Aggregate success rates from each task summaries.txt
mkdir -p "$(dirname "$SUMMARY_FILE")"
{
  echo "Robosuite CapX multiturn-VDM success rates"
  echo "Model: $MODEL"
  echo "Trials/task: $TOTAL_TRIALS  Workers: $NUM_WORKERS"
  echo "Finished: $(date -Is)"
  echo ""
  printf "%-55s %s\n" "TASK" "success/reward/completed (from summaries.txt)"
  printf "%-55s %s\n" "----" "---------------------------------------------"
  for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    # launch.py inserts model name before the final path component
    # e.g. ./outputs/foo -> ./outputs/<model>/foo
    out="outputs/${MODEL}/${stem}"
    # some yamls use ./outputs/... already; also try yaml output_dir stem
    sum=""
    for cand in \
      "outputs/${MODEL}/${stem}/summaries.txt" \
      "outputs/${MODEL}/franka_robosuite_${stem#franka_robosuite_}/summaries.txt"
    do
      if [[ -f "$cand" ]]; then sum="$cand"; break; fi
    done
    # fallback: find by stem under model dir
    if [[ -z "$sum" ]]; then
      sum="$(find "outputs/${MODEL}" -type f -path "*${stem}*/summaries.txt" 2>/dev/null | head -1 || true)"
    fi
    if [[ -n "$sum" && -f "$sum" ]]; then
      line="$(rg -n "Code generation success rate|Average reward|Task completed" -A1 "$sum" | tr '\n' ' ')"
      # Prefer the numeric triple line
      metrics="$(awk '/Code generation success rate/{getline; print; exit}' "$sum")"
      printf "%-55s %s\n" "$stem" "${metrics:-see $sum}"
    else
      printf "%-55s %s\n" "$stem" "MISSING summaries"
    fi
  done
  if ((${#failed[@]})); then
    echo ""
    echo "Failed configs: ${failed[*]}"
  fi
} | tee "$SUMMARY_FILE" | tee -a "$LOG_FILE"

echo "Summary written to $SUMMARY_FILE" | tee -a "$LOG_FILE"
if ((${#failed[@]})); then
  exit 1
fi
