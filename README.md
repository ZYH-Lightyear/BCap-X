


# CaP-X

### A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

[Project Page](https://capgym.github.io/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2603.22435) 

**Max Fu<sup>&#42;,1,2</sup>, Justin Yu<sup>&#42;,2</sup>, Karim El-Refai<sup>&#42;,2</sup>, Ethan Kou<sup>&#42;,2</sup>, Haoru Xue<sup>&#42;,1,2</sup>,
Huang Huang<sup>3</sup>, Wenli Xiao<sup>4</sup>, Guanzhi Wang<sup>1</sup>, Fei-Fei Li<sup>3</sup>, Guanya Shi<sup>4</sup>, Jiajun Wu<sup>3</sup>,
Shankar Sastry<sup>2</sup>, Yuke Zhu<sup>1</sup>, Ken Goldberg<sup>&dagger;,2</sup>, Jim Fan<sup>&dagger;,1</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>UC Berkeley &ensp; <sup>3</sup>Stanford University &ensp; <sup>4</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Equal advising



---
**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. It consists of four components:


| Component      | What it does                                                                                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CaP-Gym**    | Interactive Gymnasium environments where agents control robots by generating Python code that composes perception and control primitives. 39 tasks across Robosuite, LIBERO-PRO, and BEHAVIOR. |
| **CaP-Bench**  | Systematic benchmark evaluating coding agents across abstraction levels, interaction modes, and visual grounding modalities. 8 tiers (S1-S4 single-turn, M1-M4 multi-turn).                    |
| **CaP-Agent0** | Training-free agentic framework with multi-turn visual differencing, auto-synthesized skill libraries, and parallel ensembled reasoning.                                                       |
| **CaP-RL**     | Reinforcement learning on the coding agent via GRPO, using environment rewards to post-train language models. Transfers from sim to real with minimal gap.                                     |

### Component → code map

| Component | Primary paths |
| --------- | ------------- |
| **CaP-Gym** | `capx/envs/` + `env_configs/` |
| **CaP-Bench** | `env_configs/**` YAML tiers (S1–S4 single-turn, M1–M4 multi-turn) |
| **CaP-Agent0** | `capx/envs/trial.py`, `capx/agents/`, `capx/skills/` |
| **CaP-RL** | `capx/third_party/verl`, `verl_agent_reward/`, `scripts/train_franka_grpo.sh` |

### Repository layout

Eval data flow: `YAML → launch.py → runner.py → trial.py → LLM code → Sandbox exec → APIs → Simulator + perception servers`.

```
BCap-X/
├── capx/                        # Core framework
│   ├── envs/                    # CaP-Gym: launch / runner / trial / sandbox tasks
│   │   ├── launch.py            # CLI entry
│   │   ├── runner.py            # Parallel eval orchestration
│   │   ├── trial.py             # Single-trial loop (multi-turn visual diff)
│   │   ├── configs/             # YAML load + instantiate
│   │   ├── simulators/          # Robosuite / LIBERO / R1Pro-B1K / real Franka
│   │   ├── tasks/               # CodeExecutionEnvBase + franka / r1pro tasks
│   │   ├── assets/              # URDF / MuJoCo assets
│   │   └── scripts/             # Batch / Codex / coding-agent runners
│   ├── integrations/            # APIs exposed to LLM (docstring = interface doc)
│   │   ├── base_api.py          # ApiBase + register_api / get_api
│   │   ├── franka/              # Privileged / Full / Reduced / Skill-library APIs
│   │   ├── vision/              # SAM2/3, OWL-ViT, Molmo, Qwen point, GraspNet
│   │   ├── motion/              # PyRoKi IK, CuRobo planning
│   │   ├── r1pro/               # BEHAVIOR / R1Pro control glue
│   │   ├── libero/              # LIBERO helpers
│   │   └── robosuite/           # Controllers / robot configs
│   ├── serving/                 # Perception + LLM proxies (SAM3/GraspNet/AnyGrasp/GG-CNN/…)
│   ├── llm/                     # OpenAI-compatible LLM client
│   ├── agents/                  # Coding agent / Codex runners
│   ├── skills/                  # Skill extract + skill library
│   ├── web/                     # FastAPI / WebSocket backend for web-ui
│   ├── utils/                   # Parallel eval, camera/depth, logging, viz
│   ├── cli/                     # VeRL dataset prep, etc.
│   └── third_party/             # Git submodules (sim / perception / RL)
│       ├── LIBERO-PRO/
│       ├── robosuite/
│       ├── sam3/
│       ├── curobo/
│       ├── b1k/                 # BEHAVIOR / OmniGibson
│       ├── verl/
│       ├── contact_graspnet_pytorch/
│       ├── ggcnn/               # dougsm GG-CNN (local :8119)
│       └── …
├── env_configs/                 # CaP-Bench YAML (by task family)
│   ├── cube_stack|lifting|restack/
│   ├── nut_assembly/ spill_wipe/
│   ├── two_arm_handover|lift/
│   ├── libero/ r1pro/ real/
│   ├── human_oracle_code/
│   └── */hillclimb/             # Hill-climb experiment variants
│       # Suffix legend:
│       #   (none)              single-turn
│       #   _multiturn(_vdm|_vf) multi-turn / visual differencing / feedback
│       #   _privileged         privileged-state APIs
│       #   _reduced_api        low-level primitives
│       #   _exampleless        no in-prompt examples
│       #   _skill_lib          primitives + synthesized skills
├── robomex/                     # Multi-agent extension (planner / verifier / evolve)
│   ├── agents/ core/ perception/
│   ├── skills/ verification/
│   └── examples/
├── web-ui/                      # React / Vite interactive eval UI
├── verl_agent_reward/           # GRPO / VeRL Franka reward fns
├── scripts/                     # Regression, serve up/down, skill-lib compile, RL train
├── tests/                       # Env smoke + perception integration tests
├── docs/                        # Guides; see docs/capx-architecture.md for deep dive
├── .agents/skills/              # Reusable LIBERO agent skills
├── third_party/                 # Reference skills / tools (≠ capx/third_party)
├── capskillbench/               # Skill-benchmark placeholder
├── logs/ outputs/               # Runtime artifacts
├── pyproject.toml               # uv deps + extras (robosuite / libero / verl / …)
└── README.md
```

**Where to edit**

| Goal | Look here |
| ---- | --------- |
| Task / eval tier | `env_configs/` + `capx/envs/tasks\|simulators/` |
| Functions the model can call | `capx/integrations/` |
| Multi-turn agent loop | `capx/envs/trial.py`, `capx/agents/`, `capx/skills/` |
| Perception / LLM servers | `capx/serving/` |
| Stronger planner–verifier loop | `robomex/` |
| RL post-training | `verl_agent_reward/` + `capx/third_party/verl` + `scripts/train_franka_grpo.sh` |

---

## Installation

CaP-X uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires **Python 3.10** and a **CUDA-capable GPU**.

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x && cd cap-x

# Or if already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# Base install
uv sync
```

### Simulator-specific setup

Pick **one** simulator family to install, as Robosuite (1.5.0) and LIBERO (`robosuite==1.4.0`) would be in conflict.

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO requires a **separate virtual environment**.

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

See [docs/libero-tasks.md](docs/libero-tasks.md) for running any of 130+ LIBERO tasks.

#### BEHAVIOR (Isaac Sim)

BEHAVIOR tasks run on NVIDIA Isaac Sim via OmniGibson. Requires Python 3.10 and CUDA 12.x.

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset          # installs OmniGibson, Isaac Sim, BDDL, cuRobo, and downloads assets
cd ../../..                        # back to repo root

# Post-install fix — copy cuRobo JIT headers (run with b1k venv active)
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> The `--dataset` flag downloads robot assets, BEHAVIOR-1K scene/object assets, and 2025 challenge task instances. You will be prompted to accept the NVIDIA Isaac Sim EULA and BEHAVIOR dataset license. To auto-accept, add `--accept-dataset-tos`.

For headless servers:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
# Remove duplicate Vulkan ICD if present (causes segfault on multi-GPU systems)
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

See [docs/behavior-tasks.md](docs/behavior-tasks.md) for task details and expected baselines.

### Optional extras

```bash
uv sync --extra verl             # RL training with VeRL/GRPO
uv sync --extra contactgraspnet  # Contact-GraspNet grasp planning
uv sync --extra curobo           # cuRobo GPU-accelerated IK & motion planning (requires CUDA)
```

## Quick Start

### 1. Perception servers (auto-launched)

Perception servers (SAM3, ContactGraspNet, PyRoKi) are **auto-launched** by the YAML config when you run an evaluation. No manual setup required for most configs.

> **SAM3 authentication:** SAM3 weights require HuggingFace access. Request access at the [SAM3 repo](https://github.com/facebookresearch/sam3), then authenticate locally with `huggingface-cli login`. Weights are cached after first download.

To pre-launch servers (e.g. for sharing across multiple eval runs):

```bash
# Start SAM3 + GraspNet + PyRoKi with automatic GPU allocation
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

Use `--dry-run` to preview the allocation. Other profiles:

```bash
--profile full      # All perception servers (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2, GG-CNN)
--profile minimal   # PyRoKi only (for oracle/privileged evals)
--profile anygrasp  # AnyGrasp grasp service only (:8120; mock until license)
--profile ggcnn     # GG-CNN grasp service only (:8119)
```

**GG-CNN** optional depth-only antipodal grasps ([dougsm/ggcnn](https://github.com/dougsm/ggcnn)); vendor + Cornell weights under `capx/third_party/ggcnn/`:

```bash
python -m capx.serving.launch_ggcnn_server --device cuda --port 8119 --host 127.0.0.1
# curl http://127.0.0.1:8119/health
# from capx.integrations.vision.ggcnn import init_ggcnn
```

### 2. Set up an LLM proxy

The evaluation harness queries an LLM through a local proxy that exposes an OpenAI-compatible API.

```bash
# OpenRouter (get a key at openrouter.ai/keys)
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** `.openrouterkey` are git-ignored. The default server URL in configs is `http://127.0.0.1:8110/chat/completions`. 

See [docs/configuration.md](docs/configuration.md) for all provider options (OpenRouter, vLLM, custom).

### 3. Run evaluation

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD:$PYTHONPATH
# Robosuite: minimal (SAM3 uses remote http://101.132.143.105:6068)
conda activate sci
cd /Knowin/foundation/bohanzhou/MyProj/BCap-X
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD:$PYTHONPATH
python capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --use-oracle-code True --total-trials 5 --num-workers 1

# Robosuite: single-turn benchmark (100 trials, 12 parallel workers)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"

# Robosuite: multi-turn with visual differencing
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: spatial task (requires .venv-libero)
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro radio pickup (20 trials) — requires b1k venv
source capx/third_party/b1k/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# Interactive Web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
# Open http://localhost:8200

# Regression tests
./scripts/regression_test.sh quick    # 10-trial smoke test (~30s)
./scripts/regression_test.sh test1    # Full single-turn (~3 min)
```

> **Tip (BEHAVIOR):** Isaac Sim uses `OMNIGIBSON_GPU_ID` (not `CUDA_VISIBLE_DEVICES`) to select the GPU. For best performance on multi-GPU systems, run perception servers on a separate GPU (e.g. `OMNIGIBSON_GPU_ID=0` for the eval, and pre-launch SAM3/GraspNet with `CUDA_VISIBLE_DEVICES=1`). Set `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` for headless evaluation.

---

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, LLM provider setup |
| [LIBERO-PRO Tasks](docs/libero-tasks.md) | Setup, running any of 130+ LIBERO tasks, suite reference |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | Setup, R1Pro tasks, expected baselines, environment variables |
| [Development](docs/development.md) | Testing, linting, LIBERO/GraspNet setup, checkpoints, known issues |
| [Real-World Franka Panda Bringup](docs/real-franka.md) | Bringup with robots_realtime, real-robot QuickStart |
| [RL Training](docs/rl-training.md) | CaP-RL with GRPO/VeRL, sim-to-real transfer |
| [Skill Library Compilation](scripts/skill_library_compilation/README.md) | Analyze eval outputs, compile reusable skill libraries |

---

## Citation

```bibtex
@article{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  journal   = {arXiv preprint arXiv:2603.22435},
  year      = {2025},
  url       = {https://arxiv.org/abs/2603.22435}
}
```

## License

This project is released under the [MIT License](LICENSE).
