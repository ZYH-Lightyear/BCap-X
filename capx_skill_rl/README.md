# CaP-X Skill RL

`capx_skill_rl` is a minimal, provider-independent tool-use RL environment for
LIBERO-PRO. The policy sees only the task, the current `agentview` RGB image,
and prior tool calls/results. Depth, calibration, masks, rewards, and the
LIBERO simulator remain inside the environment.

## Action space

Every step contains exactly one action:

```json
{"name": "tool_name", "arguments": {}}
```

The complete tool list is:

```text
vlm_bbox_detection  vlm_point_detection  sam3  get_obb  plan_grasp
solve_ik  move_to_joints  open_gripper  close_gripper  go_home
```

Pixel coordinates refer to the current `agentview` RGB. All public 3D poses
use the robot-base frame and XYZW quaternions. Successful control calls return
`{}`; tool failures return `{"error": "..."}`.

## Environment contract

```python
observation = env.reset(seed)
step = env.step({"name": "go_home", "arguments": {}})
```

`reset` returns `Observation(rgb, task)`. `step` returns the tool result, an
optional refreshed RGB observation, sparse reward, and `done`. Physical tools
refresh RGB-D and invalidate every existing `mask_id`. Episodes stop on
LIBERO-PRO success or after 32 tool calls.

`run_episode` accepts any policy implementing:

```python
policy.act(task=..., rgb=..., history=..., tools=...) -> action
```

It intentionally contains no model client or RL trainer. Inference providers
and future rollout workers adapt to this protocol without changing the
environment.

## LIBERO-PRO live smoke test

Use the LIBERO environment and start the existing CaP-X VLM, SAM3, GraspNet,
and Pyroki services. Then run:

```bash
source .venv-libero/bin/activate
python -m capx_skill_rl.scripts.smoke_loop
```

The default is `libero_object_swap:0` with query `alphabet soup`. Override the
task without changing code:

```bash
python -m capx_skill_rl.scripts.smoke_loop \
  --suite-name libero_spatial_task \
  --task-id 2 \
  --query "black bowl at the table center"
```

The supported suites are the object, goal, and spatial `swap`/`task` splits.
The smoke test is perception-only and does not move the robot.

## M1 live validation

With the same services running, validate the complete ten-tool boundary:

```bash
python -m capx_skill_rl.scripts.validate_live
```

The validator exercises bbox and point detection, all three SAM3 prompt modes,
OBB and grasp planning, IK, a no-op joint move using the current arm joints,
gripper control, and home. It also verifies observation revisions and that a
mask becomes invalid after physical motion. The compact JSON report is written
to `outputs/capx_skill_rl/m1/`.

## M2 scripted episode

Run the first real pick-and-place attempt on `libero_object_swap:0`:

```bash
source .venv-libero/bin/activate
python -m capx_skill_rl.scripts.scripted_episode
```

The script uses only the public tool actions. It grounds the basket and target,
plans a grasp, executes pregrasp/grasp/lift/transfer/release as separate RL
steps, and uses the LIBERO sparse reward as the final success signal. RGB
checkpoints and the action trace are saved under `outputs/capx_skill_rl/m2/`.

## M3 model rollout

Start with a shadow rollout. It executes perception and reasoning tools, then
stops before the model's first physical action:

```bash
source .venv-libero/bin/activate
python -m capx_skill_rl.scripts.model_rollout --mode shadow
```

After inspecting the proposed action, allow a complete real rollout:

```bash
python -m capx_skill_rl.scripts.model_rollout --mode live
```

The model receives only task text, current RGB, tool schemas, and prior
action/results. Native tool calls are parsed without forced `tool_choice`;
parallel calls are disabled. IK requests are delegated directly to the CaP-X
backend; joint commands must be copied from the immediately preceding IK
result. Frames, environment transitions, latency, usage, and raw model
responses are saved under `outputs/capx_skill_rl/m3/`.

## Offline tests

```bash
pytest -q capx_skill_rl/tests
```

These tests use a fake backend and do not require LIBERO or external services.
