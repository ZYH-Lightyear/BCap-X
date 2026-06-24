# CaP SkillBench

This directory documents the CaP-X LIBERO autonomous coding-agent harness.
The implementation currently lives in `capx/agents/` and
`capx/envs/scripts/run_coding_agent_libero.py`.

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

This mirrors the SkillsBench/AgentBeats pattern: an autonomous agent process runs
inside a prepared sandbox, skills are available in the workspace, and verification
is based on artifacts rather than on the agent's final text.

## OpenCode Smoke Command

```bash
source .venv-libero/bin/activate

python capx/envs/scripts/run_coding_agent_libero.py \
  --agent opencode \
  --opencode-bin /mnt/data/zyh/bin/opencode \
  --config-path env_configs/libero/franka_libero_cap_agent0.yaml \
  --model openrouter/qwen/qwen3.6-plus \
  --skill-mode with-skill \
  --skills-dir .agents/skills \
  --total-trials 1 \
  --max-turns 3 \
  --output-dir outputs/opencode_libero_test \
  --agent-timeout-seconds 3600
```

`--max-turns` is the maximum number of simulator executions via
`tools/run_solution.py`; inspect-only runs do not count.

