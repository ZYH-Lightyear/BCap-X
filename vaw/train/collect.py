"""M4 — Teacher trace collection and SFT data building (interface only).

Pipeline:
  1. run vaw.agents.teacher over task suite x seeds -> runs/vaw/traces/<ep>/
  2. filter: env success AND no unpredicted_failure receipts
  3. build SFT samples: multi-turn interleaved image-text conversations
     (system prompt + per-step [canvas image, receipt, state summary] -> tool
     call), directly from steps.jsonl + canvas PNGs. Output: LLaMA-Factory /
     ms-swift sharegpt-style JSON with image paths.

Scale target (aligned with Skill-3D): ~500 SFT episodes, ~1k RL tasks.
"""

raise NotImplementedError("M4: implement after M3 teacher agent")
