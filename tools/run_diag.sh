#!/usr/bin/env bash
# Start LOCAL SAM3 (:8114) + GraspNet (:8115) + PyRoKi (:8116), then run the
# per-API-call diagnostic for original / improved cube-stack programs.
set -u
cd /Knowin/foundation/bohanzhou/MyProj/BCap-X
source /Knowin/foundation/bohanzhou/__backup/ENV/miniconda3/etc/profile.d/conda.sh
conda activate sci
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Local packages: pyroki + facebook/sam3 (third_party checkout).
export PYTHONPATH=$PWD:$PWD/capx/third_party/pyroki/src:$PWD/capx/third_party/sam3:$PYTHONPATH
# Local SAM3 — do NOT point at the remote DSW endpoint.
export SAM3_SERVICE_URL=http://127.0.0.1:8114
export GRASPNET_SERVICE_URL=http://127.0.0.1:8115

CKPT=/Knowin/foundation/bohanzhou/__backup/PRETRAINED/models--facebook--sam3/sam3.pt
mkdir -p /tmp/diag_logs

pkill -f launch_pyroki_server 2>/dev/null || true
pkill -f launch_contact_graspnet_server 2>/dev/null || true
pkill -f launch_sam3_server 2>/dev/null || true
sleep 2

python -m capx.serving.launch_sam3_server \
  --device cuda:0 --port 8114 --host 127.0.0.1 \
  --checkpoint-path "$CKPT" \
  > /tmp/diag_logs/sam3.log 2>&1 &
SAM_PID=$!

python -m capx.serving.launch_contact_graspnet_server \
  --port 8115 --host 127.0.0.1 \
  > /tmp/diag_logs/graspnet.log 2>&1 &
GN_PID=$!

python -m capx.serving.launch_pyroki_server \
  > /tmp/diag_logs/pyroki.log 2>&1 &
PYROKI_PID=$!

echo "sam3=$SAM_PID graspnet=$GN_PID pyroki=$PYROKI_PID"
python tools/wait_ports.py 8114 8115 8116
echo "servers ready (local SAM3)"

OUT=/Knowin/foundation/bohanzhou/MyProj/BCap-X/outputs/oracle/_diag_trial03
rm -rf "$OUT"; mkdir -p "$OUT"

echo "===== ORIGINAL ====="
python tools/diag_cube_stack.py --program original --seed 0 --outdir "$OUT/original"
echo "===== IMPROVED ====="
python tools/diag_cube_stack.py --program improved --seed 0 --outdir "$OUT/improved"

# Also dump the improved program as a plain oracle-style code.py for comparison.
python - <<'PY'
from pathlib import Path
code = r'''import numpy as np
from scipy.spatial.transform import Rotation as Rt

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
red_pos_meas, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Keep Contact-GraspNet pose when it is close to the cube center; otherwise
# fall back to a top-down grasp at the measured cube center (trial_03 failure mode).
xy_err = float(np.linalg.norm(grasp_pos[:2] - red_pos_meas[:2]))
if xy_err > 0.02:
    yaw = Rt.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]).as_euler("xyz")[2]
    q_xyzw = Rt.from_euler("xyz", [np.pi, 0.0, yaw]).as_quat()
    pick_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    pick_pos = red_pos_meas.copy()
else:
    pick_pos, pick_quat = grasp_pos, grasp_quat

goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()

post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.20
goto_pose(post_pick_pos, pick_quat)

green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)
place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2] / 2 + red_ext[2] / 2

goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, pick_quat)
'''
out = Path("/Knowin/foundation/bohanzhou/MyProj/BCap-X/outputs/oracle/_diag_trial03/improved_code.py")
out.write_text(code)
trial = Path("/Knowin/foundation/bohanzhou/MyProj/BCap-X/outputs/oracle/franka_robosuite_cube_stack_main_use_server/trial_03_sandboxrc_0_reward_0.002_taskcompleted_0")
(trial / "code_improved.py").write_text(code)
print("wrote", out)
print("wrote", trial / "code_improved.py")
PY

kill "$SAM_PID" "$GN_PID" "$PYROKI_PID" 2>/dev/null || true
echo DONE
ls -R "$OUT" | head -80
