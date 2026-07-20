from __future__ import annotations

from pathlib import Path

import pytest

from robomex.authoring.artifacts import (
    ArtifactStore,
    PortSpec,
    TypedArtifact,
    add_grounding_overlay,
    outputs_from_finish,
)
from robomex.authoring.capabilities import (
    PERCEPTION_READ,
    CapabilityPolicy,
    called_function_names,
)
from robomex.core.context import ArtifactRef
from robomex.core.sandbox import RuntimeSafetyState


def _artifact(
    producer: str,
    port: str,
    schema: str = "test.value.v1",
    *,
    epoch: int = 0,
    frame: str = "",
) -> TypedArtifact:
    return TypedArtifact(
        producer=producer,
        port=port,
        schema=schema,
        payload={"value": producer},
        observation_epoch=epoch,
        frame=frame,
    )


def test_store_requires_qualified_explicit_refs() -> None:
    store = ArtifactStore()
    store.publish("ground", (PortSpec("mask", "test.value.v1"),), (_artifact("ground", "mask"),))

    assert store.resolve(
        (PortSpec("input_mask", "test.value.v1"),),
        {"input_mask": {"$ref": "ground.mask"}},
    )["input_mask"].producer == "ground"
    with pytest.raises(ValueError, match="producer.port"):
        store.resolve(
            (PortSpec("input_mask", "test.value.v1"),),
            {"input_mask": {"$ref": "mask"}},
        )


def test_store_validates_schema_frame_and_epoch() -> None:
    state = RuntimeSafetyState(observation_epoch=2)
    store = ArtifactStore(epoch_source=state)
    store.publish(
        "camera",
        (PortSpec("pose", "test.pose.v1", frame="world"),),
        (_artifact("camera", "pose", "test.pose.v1", epoch=2, frame="world"),),
    )
    with pytest.raises(ValueError, match="schema"):
        store.resolve(
            (PortSpec("pose", "other.v1"),), {"pose": {"$ref": "camera.pose"}}
        )
    with pytest.raises(ValueError, match="frame"):
        store.resolve(
            (PortSpec("pose", "test.pose.v1", frame="camera"),),
            {"pose": {"$ref": "camera.pose"}},
        )
    state.observation_epoch = 3
    with pytest.raises(ValueError, match="stale"):
        store.resolve(
            (PortSpec("pose", "test.pose.v1"),), {"pose": {"$ref": "camera.pose"}}
        )


def test_catalog_marks_and_can_hide_stale_artifacts() -> None:
    state = RuntimeSafetyState(observation_epoch=1)
    store = ArtifactStore(epoch_source=state)
    store.publish(
        "camera",
        (PortSpec("pose", "test.pose.v1"),),
        (_artifact("camera", "pose", "test.pose.v1", epoch=1),),
    )

    assert store.catalog()[0]["stale"] is False
    state.observation_epoch = 2

    assert store.catalog()[0]["stale"] is True
    assert store.catalog(include_stale=False) == []


def test_publish_is_atomic_when_one_output_is_invalid(tmp_path: Path) -> None:
    store = ArtifactStore(artifact_root=tmp_path)
    valid = _artifact("node", "valid")
    invalid = TypedArtifact(
        producer="node",
        port="file",
        schema="test.file.v1",
        refs=(ArtifactRef("missing", path="missing.bin"),),
    )
    with pytest.raises(ValueError, match="missing file"):
        store.publish(
            "node",
            (
                PortSpec("valid", "test.value.v1"),
                PortSpec("file", "test.file.v1"),
            ),
            (valid, invalid),
        )
    assert store.values() == ()


def test_finish_parser_stamps_runtime_epoch_over_agent_echo() -> None:
    """M1.5 Fix A: an echoed request-time epoch must not mark fresh evidence stale."""

    outputs = outputs_from_finish(
        {
            "outputs": {
                "evidence": {
                    "payload": {"all_ok": True},
                    # Agent echoes the epoch it saw at request time, before its own
                    # motion advanced the store epoch.
                    "observation_epoch": 1,
                }
            }
        },
        producer="execute",
        ports=(PortSpec("evidence", "test.value.v1"),),
        observation_epoch=2,
        artifact_dir=None,
    )
    assert outputs[0].observation_epoch == 2

    state = RuntimeSafetyState(observation_epoch=2)
    store = ArtifactStore(epoch_source=state)
    store.publish("execute", (PortSpec("evidence", "test.value.v1"),), outputs)
    assert store.values()[0].observation_epoch == 2


def test_finish_parser_rejects_missing_required_output() -> None:
    with pytest.raises(ValueError, match="omitted required"):
        outputs_from_finish(
            {"ok": True, "outputs": {}},
            producer="leaf",
            ports=(PortSpec("pose", "test.pose.v1"),),
            observation_epoch=0,
            artifact_dir=None,
        )


def test_finish_parser_accepts_frame_and_promotes_payload_paths(tmp_path: Path) -> None:
    points_path = tmp_path / "points.npy"
    points_path.write_bytes(b"data")
    outputs = outputs_from_finish(
        {
            "outputs": {
                "points": {
                    "payload": {
                        "frame": "world",
                        "center": [0.1, 0.2, 0.3],
                        "points_path": str(points_path),
                    },
                }
            }
        },
        producer="ground",
        ports=(PortSpec("points", "robomex.points3d.v1", frame="world"),),
        observation_epoch=0,
        artifact_dir=tmp_path,
    )

    store = ArtifactStore()
    store.publish(
        "ground",
        (PortSpec("points", "robomex.points3d.v1", frame="world"),),
        outputs,
    )
    assert outputs[0].frame == "world"
    assert "points_path" not in outputs[0].payload
    assert outputs[0].refs[0].path == str(points_path)


def test_finish_parser_normalizes_rendered_port_label() -> None:
    outputs = outputs_from_finish(
        {
            "outputs": {
                "trajectory:robomex.trajectory.v1@world": {
                    "payload": {"waypoints": []},
                    "frame": "world",
                }
            }
        },
        producer="planner",
        ports=(PortSpec("trajectory", "robomex.trajectory.v1", frame="world"),),
        observation_epoch=0,
        artifact_dir=None,
    )

    assert outputs[0].port == "trajectory"


def test_finish_artifact_paths_generate_grounding_overlay(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rgb_path = tmp_path / "rgb.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(rgb_path, np.zeros((8, 10, 3), dtype=np.uint8))
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    np.save(mask_path, mask)

    outputs = outputs_from_finish(
        {
            "outputs": {
                "mask": {
                    "payload": {"format": "boolean_mask_npy", "bbox_xyxy": [3, 2, 7, 5]},
                    "artifact_path": str(mask_path),
                },
                "rgb": {
                    "payload": {"format": "uint8_rgb_npy"},
                    "artifact_path": str(rgb_path),
                },
            }
        },
        producer="grounder",
        ports=(
            PortSpec("mask", "robomex.mask.v1"),
            PortSpec("rgb", "robomex.rgb.v1"),
        ),
        observation_epoch=0,
        artifact_dir=tmp_path,
    )
    outputs = add_grounding_overlay(outputs, artifact_dir=tmp_path)

    assert (tmp_path / "grounding_overlay.png").is_file()
    assert any(ref.kind == "overlay" for ref in outputs[0].refs)


def test_capability_policy_uses_qualified_calls_and_allows_pure_compute() -> None:
    code = "x = np.asarray(values)\nok = np.isfinite(x).all()\np = os.path.join('a', 'b')"
    assert called_function_names(code) == (
        "all",
        "np.asarray",
        "np.isfinite",
        "os.path.join",
    )
    assert CapabilityPolicy().violations(code) == ()
    assert CapabilityPolicy().violations("goto_pose(target)") == (
        ("goto_pose", "robot_motion"),
    )
    assert CapabilityPolicy().violations("eval(source)") == (("eval", "unknown"),)
    assert CapabilityPolicy().violations("x = (") == (("<syntax>", "invalid_syntax"),)


def test_capability_policy_never_blocks_introspection_builtins() -> None:
    """M1.5 Fix B: getattr/vars/dir are pure in-process reads, not a boundary."""

    code = (
        "v = getattr(q, 'external', None)\n"
        "d = vars(q)\n"
        "names = dir(q)\n"
        "setattr(q, 'seen', True)\n"
        "ok = isinstance(q, object) and hasattr(q, 'wrist')"
    )
    assert CapabilityPolicy().violations(code) == ()
    # Even the strictest policy (deny unknown calls) treats them as plain Python.
    strict = CapabilityPolicy(unknown_calls="deny")
    assert ("getattr", "unknown") not in strict.violations(code)

    # Process escapes and dynamic code execution stay hard-denied for everyone.
    assert CapabilityPolicy().violations("subprocess.run(['ls'])") == (
        ("subprocess.run", "unknown"),
    )


def test_capability_denial_message_names_granted_apis() -> None:
    """M1.5 Fix B: a denial is a repair hint, not a bare 'blocked'."""

    from robomex.authoring.capabilities import render_denial_message

    policy = CapabilityPolicy(allowed=frozenset({PERCEPTION_READ}))
    message = render_denial_message((("goto_pose", "robot_motion"),), policy)
    assert "goto_pose" in message
    assert "robot_motion" in message
    assert "perception_read" in message  # granted set is listed
    assert "get_observation" in message  # concrete usable API is listed
