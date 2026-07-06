# CaP SkillBench

CaP SkillBench is the CaP-X autonomous coding-agent harness for manipulation
skills. It mirrors the SkillsBench/AgentBeats pattern: each benchmark trial
materializes a self-contained workspace, launches an agent-under-test inside
that workspace, and scores the run from artifacts produced by the environment
rather than from the agent's final text.

The benchmark harness now lives in this package:

```text
capskillbench/
  agents/runner.py        # Codex/OpenCode CLI harness adapters
  workspace.py            # generic workspace exports
  libero_workspace.py     # LIBERO profile workspace/task/runner renderer
  unified_api.py          # agent-facing API facade over backend CapX APIs
  skills/                 # env-agnostic agent skills copied into trials
  scripts/run.py          # profile-dispatching benchmark CLI
  scripts/run_libero.py   # LIBERO profile CLI
```

Compatibility wrappers remain under `capx/agents/` and
`capx/envs/scripts/run_codex_libero.py`, but new development should target
`capskillbench`.

## Design

Each trial creates an isolated workspace:

```text
workspace/
  task.md
  api_contract.md
  solution.py
  opencode.json
  tools/run_solution.py
  .agents/skills/
  artifacts/
  logs/
```

The outer harness only prepares the workspace, starts CaP-X API servers, launches
the coding-agent CLI, and records the final result. The coding agent is expected
to inspect the task, edit `solution.py`, run `tools/run_solution.py`, read
`artifacts/result.json` and stderr/stdout/images, then continue editing and
rerunning until success or the simulation-run budget is exhausted.

The layout intentionally resembles SkillsBench native task packages:

```text
task package / prepared workspace
  task.md                 # task description and inspect command
  api_contract.md         # callable environment API surface
  solution.py             # agent-owned policy file
  tools/run_solution.py   # verifier/executor bridge into CapX
  .agents/skills/         # optional injected skills
  artifacts/              # result.json, images, stdout/stderr, videos
  logs/                   # raw agent logs
```

The outer harness prepares this workspace, starts CaP-X API servers, runs the
agent CLI, and records a summary. The agent is expected to inspect the task,
edit `solution.py`, run `tools/run_solution.py`, read `artifacts/result.json`
and stderr/stdout/images, then continue editing and rerunning until success or
the simulation-run budget is exhausted.

`solution.py` should use the unified agent-facing API documented in
`api_contract.md` (`get_camera`, `segment_object`, `ground_point`, `goto_pose`,
`open_gripper`, `close_gripper`, etc.). The runner installs this facade before
executing the solution and routes each function to the backend CapX API selected
by the trial config. Backend-specific helpers remain available only as a debug
escape hatch.

The shipped skills in `capskillbench/skills/` are environment-agnostic. They
describe manipulation concepts such as `$scene-observation`, `$grasp-object`,
`$place-and-release`, and `$motion-control` using only the unified API. The
legacy root `.agents/skills/libero-*` files may still exist for compatibility,
but new benchmark work should prefer `capskillbench/skills`.

Each trial also gets an isolated `agent_home/` for agent-side state. OpenCode
uses it as `HOME` plus XDG cache/config/data directories. Codex uses
`agent_home/.codex` as `CODEX_HOME` and copies only the current `config.toml`;
API keys remain environment variables so benchmark outputs do not persist
secrets.

## OpenCode Smoke Command

```bash
source .venv-libero/bin/activate

python -m capskillbench.scripts.run \
  --env-profile libero -- \
  --agent opencode \
  --opencode-bin /mnt/data/zyh/bin/opencode \
  --config-path env_configs/libero/franka_libero_cap_agent0.yaml \
  --model openrouter/qwen/qwen3.6-plus \
  --skill-mode with-skill \
  --skills-dir capskillbench/skills \
  --total-trials 1 \
  --max-turns 3 \
  --output-dir outputs/opencode_libero_test \
  --agent-timeout-seconds 3600 \
  --point-backend auto \
  --vlm-model openrouter/qwen/qwen3.6-plus
```

`--max-turns` is the maximum number of simulator executions via
`tools/run_solution.py`; inspect-only runs do not count.

`--point-backend auto` configures the backend used by the unified
`ground_point(...)` helper. The historical `point_prompt_molmo(...)` backend API
may still exist as an escape hatch, but benchmark solutions should prefer the
unified helper.

The old entry point still works for compatibility:

```bash
python -m capskillbench.scripts.run_libero ...
python capx/envs/scripts/run_coding_agent_libero.py ...
```

## Claude Code Smoke Command

Claude Code support uses print mode with stream JSON logs and an isolated
per-trial `agent_home/`. For V-API, export `V_API_KEY`; the runner maps it to
Claude Code's `ANTHROPIC_AUTH_TOKEN` and sets `ANTHROPIC_BASE_URL` to
`https://api.v3.cm` inside the trial process.

```bash
source .venv-libero/bin/activate

python -m capskillbench.scripts.run \
  --env-profile libero -- \
  --agent claude \
  --claude-bin claude \
  --config-path env_configs/libero/franka_libero_cap_agent0.yaml \
  --model claude-sonnet-4-6 \
  --skill-mode with-skill \
  --skills-dir capskillbench/skills \
  --total-trials 1 \
  --max-turns 3 \
  --output-dir outputs/claude_libero_test \
  --agent-timeout-seconds 3600 \
  --agent-retries 1 \
  --claude-max-turns 60 \
  --point-backend auto \
  --vlm-model openrouter/qwen/qwen3.6-plus \
  --stream-agent-logs
```
