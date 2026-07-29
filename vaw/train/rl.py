"""M5 — Multi-turn RL entry point (interface only; built on verl).

Plan: use verl's multi-turn agent-loop rollout (sglang/vllm server mode) with
the Workspace as the environment:

    class VAWEnv:                      # adapts Workspace to verl's agent loop
        def reset(task) -> obs:        # obs = canvas image + state summary
        def step(action_text) -> (obs, reward_parts, done):
            op, args = protocol.parse_action(action_text)
            result = workspace.step(op, **args)
            ...

Rewards from vaw.train.rewards, combined per docs plan §3.3. Recipe knobs
(from ARPO / Skill-3D / UI-TARS-2 findings):
  - task filtering: keep tasks with >=1 success in 16 base-policy rollouts
  - SFT cold start mandatory; GRPO first, switch to PPO if reward variance blows up
  - freeze tools, controller and the TOPReward scorer during RL
"""

raise NotImplementedError("M5: implement after M4 SFT baseline")
