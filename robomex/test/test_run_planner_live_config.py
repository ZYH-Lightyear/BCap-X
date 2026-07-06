from __future__ import annotations

import os

from robomex.examples.run_planner_live import (
    LiveArgs,
    _api_docs,
    _configure_vlm_backend,
    _robomex_runtime_config,
    _robomex_vlm_config,
    _runtime_settings,
    _save_scene_image,
)
from robomex.examples.run_planner_batch import BatchArgs, _dump_batch_summary, _to_live_args


def test_default_libero_config_declares_swarm_runtime() -> None:
    cfg = "env_configs/libero/franka_libero_cap_agent0.yaml"

    parsed = _robomex_runtime_config(cfg)
    settings = _runtime_settings(LiveArgs(config_path=cfg))

    assert parsed["max_subgoals"] == 8
    assert parsed["subagent_max_turns"] == 8
    assert settings.scene_camera == "agentview"
    assert settings.act_observation_camera == "agentview"
    assert "get_observation" in settings.prompt_api_names
    assert "query_vlm" in settings.prompt_api_names
    assert "vlm_bbox_detection" in settings.prompt_api_names
    assert "segment_sam3_box_prompt" in settings.prompt_api_names
    assert "plan_grasp" in settings.prompt_api_names
    assert "goto_pose" in settings.subagent_denied_calls
    assert "open_gripper" in settings.subagent_denied_calls


def test_api_docs_group_core_and_capability_apis() -> None:
    class Api:
        def functions(self):
            return {
                "get_observation": self.get_observation,
                "query_vlm": self.query_vlm,
                "vlm_bbox_detection": self.vlm_bbox_detection,
                "parse_vlm_detections": self.parse_vlm_detections,
            }

        def get_observation(self):
            """Get obs."""

        def query_vlm(self, prompt, images=None):
            """Raw implementation doc should be overridden."""

        def vlm_bbox_detection(self, rgb, target):
            """Get a box."""

        def parse_vlm_detections(self, reply):
            """Parse raw VLM output."""

    class Env:
        _apis = {"api": Api()}

    docs = _api_docs(
        Env(),
        frozenset({"get_observation", "query_vlm", "vlm_bbox_detection", "parse_vlm_detections"}),
    )

    assert "Core APIs" in docs
    assert "get_observation()" in docs
    assert "query_vlm(prompt, images=None" in docs
    assert "Canonical form: query_vlm('question', images=rgb_or_crop)" in docs
    assert "Raw implementation doc" not in docs
    assert "Capability APIs" in docs
    assert "vlm_bbox_detection(rgb, target)" in docs
    assert "Advanced/internal APIs" in docs
    assert "parse_vlm_detections(reply)" in docs


def test_vlm_backend_config_comes_from_yaml(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  vlm:
    model: vapi/gpt-5.5
    server_url: http://localhost:8110/chat/completions
    coord_space: pixel
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("CAPX_VLM_MODEL", raising=False)
    monkeypatch.delenv("CAPX_VLM_SERVER_URL", raising=False)
    monkeypatch.delenv("CAPX_VLM_COORD_SPACE", raising=False)

    parsed = _robomex_vlm_config(str(cfg))
    applied = _configure_vlm_backend(
        LiveArgs(
            config_path=str(cfg),
            model="vapi/claude-opus-4.8",
            server_url="http://localhost:9999/chat/completions",
        )
    )

    assert parsed["model"] == "vapi/gpt-5.5"
    assert applied["model"] == "vapi/gpt-5.5"
    assert applied["server_url"] == "http://localhost:8110/chat/completions"
    assert applied["coord_space"] == "pixel"
    assert "CAPX_VLM_MODEL" not in os.environ
    assert "CAPX_VLM_SERVER_URL" not in os.environ
    assert "CAPX_VLM_COORD_SPACE" not in os.environ


def test_vlm_backend_falls_back_to_run_model_without_yaml_config(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("env:\n  _target_: unused\n", encoding="utf-8")
    monkeypatch.delenv("CAPX_VLM_MODEL", raising=False)
    monkeypatch.delenv("CAPX_VLM_SERVER_URL", raising=False)

    applied = _configure_vlm_backend(
        LiveArgs(
            config_path=str(cfg),
            model="vapi/claude-opus-4.8",
            server_url="http://localhost:8110/chat/completions",
        )
    )

    assert applied["model"] == "vapi/claude-opus-4.8"
    assert applied["server_url"] == "http://localhost:8110/chat/completions"
    assert "CAPX_VLM_MODEL" not in os.environ
    assert "CAPX_VLM_SERVER_URL" not in os.environ


def test_capx_executor_adapter_applies_vlm_backend_to_api_instances() -> None:
    from robomex.core.sandbox import CapXExecutorAdapter

    class Api:
        def __init__(self) -> None:
            self.configured = None

        def configure_vlm_backend(self, **kwargs) -> None:
            self.configured = kwargs

    class Env:
        def __init__(self) -> None:
            self.api = Api()
            self._apis = {"libero": self.api}

    env = Env()
    CapXExecutorAdapter(
        env,
        vlm_backend={
            "model": "vapi/gpt-5.5",
            "server_url": "http://localhost:8110/chat/completions",
            "coord_space": "pixel",
        },
    )

    assert env.api.configured == {
        "model": "vapi/gpt-5.5",
        "server_url": "http://localhost:8110/chat/completions",
        "coord_space": "pixel",
    }


def test_capx_executor_adapter_preserves_video_range_probe() -> None:
    from robomex.core.sandbox import CapXExecutorAdapter, SemanticActionBlock

    class Env:
        def __init__(self) -> None:
            self.frames = 2

        def get_video_frame_count(self) -> int:
            return self.frames

        def step(self, _code):
            self.frames = 5
            return {}, 0.0, False, False, {"sandbox_rc": 0}

    result = CapXExecutorAdapter(Env()).run_block(
        SemanticActionBlock(name="turn_0", code="print('x')", intent="test")
    )

    assert result.info["video_range"] == (2, 5)


def test_capx_executor_adapter_records_api_primitive_traces() -> None:
    from robomex.core.sandbox import CapXExecutorAdapter, SemanticActionBlock

    class Api:
        def functions(self):
            return {
                "vlm_bbox_detection": self.vlm_bbox_detection,
                "goto_pose": self.goto_pose,
                "untracked_helper": self.untracked_helper,
            }

        def vlm_bbox_detection(self, _rgb, target):
            return [1, 2, 3, 4] if target else [0, 0, 0, 0]

        def goto_pose(self, position, quat):
            return None

        def untracked_helper(self):
            return "ok"

    class Env:
        def __init__(self) -> None:
            self._apis = {"fake": Api()}
            self._exec_globals = {}
            self._events = []

        def configure_line_trace(self, _block_index):
            self._events = []

        def consume_line_trace_events(self):
            events = list(self._events)
            self._events = []
            return events

        def step(self, _code):
            for api in self._apis.values():
                for name, fn in api.functions().items():
                    self._exec_globals[name] = fn
            box = self._exec_globals["vlm_bbox_detection"]("rgb", "can")
            self._exec_globals["goto_pose"]([0, 0, 0], [0, 1, 0, 0])
            self._exec_globals["untracked_helper"]()
            return {}, 0.0, False, False, {"sandbox_rc": 0, "stdout": str(box)}

    result = CapXExecutorAdapter(Env()).run_block(
        SemanticActionBlock(name="turn_0", code="fake", intent="test")
    )

    traces = []
    for event in result.trace_events:
        traces.extend(event.payload.get("primitive_traces") or [])
    names = [trace["primitive_name"] for trace in traces]

    assert names == ["vlm_bbox_detection", "goto_pose"]
    assert traces[0]["outputs_summary"]["result"] == [1, 2, 3, 4]
    assert traces[0]["status"] == "succeeded"
    assert result.stdout == "[1, 2, 3, 4]"


def test_vlm_backend_ignores_removed_top_level_alias(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
vlm:
  model: should-not-be-used
""",
        encoding="utf-8",
    )

    assert _robomex_vlm_config(str(cfg)) == {}


def test_runtime_config_comes_from_robomex_runtime(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  runtime:
    max_subgoals: 4
    subagent_max_turns: 7
    scene_camera: wrist
    act_observation_camera: robot0_eye_in_hand
    prompt_api_names:
      - get_observation
      - query_vlm
    subagent_denied_calls:
      - goto_pose
      - custom_motion
""",
        encoding="utf-8",
    )

    parsed = _robomex_runtime_config(str(cfg))
    settings = _runtime_settings(LiveArgs(config_path=str(cfg)))

    assert parsed["max_subgoals"] == 4
    assert settings.max_subgoals == 4
    assert settings.subagent_max_turns == 7
    assert settings.scene_camera == "wrist"
    assert settings.act_observation_camera == "robot0_eye_in_hand"
    assert settings.prompt_api_names == frozenset({"get_observation", "query_vlm"})
    assert settings.subagent_denied_calls == frozenset({"goto_pose", "custom_motion"})


def test_runtime_camera_defaults_to_auto_without_config(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("env:\n  _target_: unused\n", encoding="utf-8")

    settings = _runtime_settings(LiveArgs(config_path=str(cfg)))

    assert settings.scene_camera == ""
    assert settings.act_observation_camera == ""


def test_runtime_cli_values_override_yaml_defaults(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  runtime:
    max_subgoals: 4
    subagent_max_turns: 7
    scene_camera: wrist
""",
        encoding="utf-8",
    )

    settings = _runtime_settings(
        LiveArgs(
            config_path=str(cfg),
            max_subgoals=9,
            subagent_max_turns=3,
            scene_camera="frontview",
        )
    )

    assert settings.max_subgoals == 9
    assert settings.subagent_max_turns == 3
    assert settings.scene_camera == "frontview"
    assert settings.act_observation_camera == "frontview"


def test_save_scene_image_auto_selects_first_rgb_camera(tmp_path) -> None:
    import numpy as np

    obs = {
        "state": {"qpos": [0.0]},
        "wrist": {"images": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}},
    }

    out = _save_scene_image(obs, str(tmp_path / "scene.png"), camera="")

    assert out is not None
    assert (tmp_path / "scene.png").exists()


def test_runtime_act_observation_camera_can_override_scene_camera(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  runtime:
    scene_camera: agentview
    act_observation_camera: robot0_eye_in_hand
""",
        encoding="utf-8",
    )

    settings = _runtime_settings(LiveArgs(config_path=str(cfg)))

    assert settings.scene_camera == "agentview"
    assert settings.act_observation_camera == "robot0_eye_in_hand"


def test_runtime_cli_scene_camera_override_updates_act_camera_when_act_not_explicit(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  runtime:
    scene_camera: agentview
    act_observation_camera: agentview
""",
        encoding="utf-8",
    )

    settings = _runtime_settings(LiveArgs(config_path=str(cfg), scene_camera="frontview"))

    assert settings.scene_camera == "frontview"
    assert settings.act_observation_camera == "frontview"


def test_batch_args_forward_swarm_runtime_to_live_args() -> None:
    args = BatchArgs(
        config_path="config.yaml",
        model="vapi/gpt-5.5",
        server_url="http://localhost:8110/chat/completions",
        max_turns=9,
        max_subgoals=4,
        subagent_max_turns=7,
        scene_camera="frontview",
        act_observation_camera="robot0_eye_in_hand",
        output_dir="outputs/test",
    )

    live = _to_live_args(args)

    assert live.max_turns == 9
    assert live.max_subgoals == 4
    assert live.subagent_max_turns == 7
    assert live.scene_camera == "frontview"
    assert live.act_observation_camera == "robot0_eye_in_hand"


def test_batch_summary_records_swarm_runtime(tmp_path) -> None:
    args = BatchArgs(
        max_turns=9,
        max_subgoals=4,
        subagent_max_turns=7,
        scene_camera="frontview",
        act_observation_camera="",
    )

    summary = _dump_batch_summary(
        tmp_path,
        args,
        suite="libero_object_swap",
        task_ids=[0],
        all_records=[],
        per_task=[],
        elapsed=1.2,
    )

    runtime = summary["runtime"]
    assert runtime["max_turns"] == 9
    assert runtime["max_subgoals"] == 4
    assert runtime["subagent_max_turns"] == 7
    assert runtime["scene_camera"] == "frontview"
    assert runtime["act_observation_camera"] == "frontview"
    assert (tmp_path / "batch_summary.json").exists()


def test_batch_summary_records_effective_yaml_runtime(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: unused
robomex:
  runtime:
    max_subgoals: 5
    subagent_max_turns: 9
    scene_camera: wrist
    act_observation_camera: robot0_eye_in_hand
    prompt_api_names:
      - get_observation
      - query_vlm
    subagent_denied_calls:
      - goto_pose
      - open_gripper
""",
        encoding="utf-8",
    )
    args = BatchArgs(config_path=str(cfg))

    summary = _dump_batch_summary(
        tmp_path,
        args,
        suite="libero_object_swap",
        task_ids=[0],
        all_records=[],
        per_task=[],
        elapsed=1.2,
    )

    runtime = summary["runtime"]
    assert runtime["max_subgoals"] == 5
    assert runtime["subagent_max_turns"] == 9
    assert runtime["scene_camera"] == "wrist"
    assert runtime["act_observation_camera"] == "robot0_eye_in_hand"
    assert runtime["prompt_api_names"] == ["get_observation", "query_vlm"]
    assert runtime["subagent_denied_calls"] == ["goto_pose", "open_gripper"]
