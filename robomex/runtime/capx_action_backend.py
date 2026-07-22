"""Trusted RoboMEx v2 action adapter for Cap-X/LIBERO.

This module is intentionally separate from the coding-agent ``CapXExecutorAdapter``.
The latter evaluates model-written Python and exposes pose/IK helpers; this adapter is
the sealed runtime boundary and exposes only three closed effects:

* execute the exact admitted arm joint path,
* set one gripper-width command, and
* hold the current controller command for a bounded interval.

No method in this module accepts a Cartesian pose and no fallback calls ``solve_ik`` or
``goto_pose``.  Snapshot revisions and semantic configuration/collision digests are
also never inferred from a mutable environment: callers must inject an explicit,
authoritative provider.

Cap-X's current LIBERO control API provides blocking waypoint control but has no public
per-control-step callback. :class:`LiberoControlPort` therefore uses the simulator's
existing ``_tracking_step`` surface when it is available, reproducing its minimum-jerk
blocking controller while exposing cooperative progress samples.  On a compatible
controller without that surface, interruption remains safe at waypoint boundaries.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from robomex.data.embodied_state import AttachmentStatus
from robomex.runtime.action_protocol import (
    ActionSpec,
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ControllerState,
    FeasibilityCertificate,
    FeasibilityStatus,
    GripperCommand,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
    MotionPlan,
    WaitSpec,
    WorldKind,
    validate_action_spec,
)
from robomex.runtime.authority import build_feasibility_certificate

JsonScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JsonValue: TypeAlias = (  # noqa: UP040
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
)
ProgressCallback: TypeAlias = Callable[  # noqa: UP040
    [Mapping[str, object]], bool
]

_CONTROL_ALWAYS_SIGNALS = frozenset(
    {"joint_positions_rad", "cooperative_granularity"}
)
_MONITOR_RESERVED_SIGNALS = frozenset(
    {
        *_CONTROL_ALWAYS_SIGNALS,
        "waypoint_index",
        "control_step",
        "joint_target_rad",
        "joint_error_rad",
        "elapsed_s",
        "settling",
        "controller_status",
        "phase",
        "sequence",
        "world_id",
        "resource_id",
    }
)


class CapXActionBackendError(RuntimeError):
    """Base error for a rejected or failed trusted adapter operation."""


class CapXBackendConfigurationError(CapXActionBackendError, ValueError):
    """The injected environment/provider cannot uphold the sealed contract."""


class CapXUnsupportedEffectError(CapXActionBackendError):
    """A sealed effect requests controller semantics unavailable on this backend."""


class ResourceRole(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    ARM = "arm"
    GRIPPER = "gripper"
    CONTROLLER = "controller"


@dataclass(frozen=True)
class SnapshotRevisions:
    """Monotonic physical revisions supplied by the authoritative state owner."""

    robot: int
    scene: int
    attachment: int
    config: int

    def __post_init__(self) -> None:
        if min(self.robot, self.scene, self.attachment, self.config) < 0:
            raise ValueError("snapshot revisions must be non-negative")


@dataclass(frozen=True)
class JointState:
    """Ordered arm state in radians; gripper state must not be appended here."""

    names: tuple[str, ...]
    positions_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.positions_rad):
            raise ValueError("joint names and positions must have equal non-zero width")
        if len(set(self.names)) != len(self.names):
            raise ValueError("joint names must be unique and ordered")
        if not all(name.strip() for name in self.names):
            raise ValueError("joint names must not be empty")
        if not all(math.isfinite(value) for value in self.positions_rad):
            raise ValueError("joint positions must be finite")


@runtime_checkable
class AdmissionSnapshotProvider(Protocol):
    """Explicit authoritative provider used at both admission and execution gates."""

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot: ...


class CallbackAdmissionSnapshotProvider:
    """Build strict admission snapshots from explicit physical-state callbacks.

    The provider deliberately has no defaults for revisions or digests.  A reset,
    controller reconfiguration, collision-world rebuild, or attachment update must be
    represented by the callback owner; otherwise a stale sealed action cannot be
    distinguished from the current world.
    """

    def __init__(
        self,
        *,
        joint_state: Callable[[str, str], JointState],
        revisions: Callable[[str, str], SnapshotRevisions],
        config_digest: Callable[[str, str], str],
        collision_world_digest: Callable[[str, str], str],
        world_kind: Callable[[str], WorldKind | str],
        attachment_status: Callable[[str, str], AttachmentStatus | str],
        controller_state: Callable[[str, str], ControllerState | str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._joint_state = joint_state
        self._revisions = revisions
        self._config_digest = config_digest
        self._collision_world_digest = collision_world_digest
        self._world_kind = world_kind
        self._attachment_status = attachment_status
        self._controller_state = controller_state
        self._clock = clock or (lambda: datetime.now(timezone.utc))  # noqa: UP017

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        state = self._joint_state(world_id, resource_id)
        revisions = self._revisions(world_id, resource_id)
        captured_at = self._clock()
        return AdmissionSnapshot(
            world_id=world_id,
            world_kind=WorldKind(self._world_kind(world_id)),
            resource_id=resource_id,
            robot_revision=revisions.robot,
            scene_revision=revisions.scene,
            attachment_revision=revisions.attachment,
            config_revision=revisions.config,
            joint_names=state.names,
            joint_positions_rad=state.positions_rad,
            config_digest=self._config_digest(world_id, resource_id),
            collision_world_digest=self._collision_world_digest(world_id, resource_id),
            attachment_status=AttachmentStatus(
                self._attachment_status(world_id, resource_id)
            ),
            controller_state=ControllerState(
                self._controller_state(world_id, resource_id)
            ),
            captured_at=captured_at,
        )


@dataclass(frozen=True)
class ControllerMoveResult:
    """Closed result from one exact waypoint target."""

    converged: bool
    timed_out: bool = False
    interrupted: bool = False
    stalled: bool = False
    telemetry: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.interrupted and self.timed_out:
            raise ValueError("a move cannot be interrupted and timed out")
        if self.converged and (self.interrupted or self.timed_out):
            raise ValueError("an interrupted/timed-out move cannot be converged")


@runtime_checkable
class CapXControlPort(Protocol):
    """Narrow physical control surface required by the sealed adapter."""

    @property
    def control_frequency_hz(self) -> float: ...

    @property
    def max_gripper_width_m(self) -> float: ...

    @property
    def monitor_signal_names(self) -> tuple[str, ...]: ...

    @property
    def monitor_supported_hooks(self) -> tuple[MonitorTelemetryHook, ...]: ...

    @property
    def cooperative_stop_guaranteed(self) -> bool: ...

    @property
    def monitor_bindings(self) -> frozenset[tuple[str, str]]: ...

    def current_joint_positions(self) -> tuple[float, ...]: ...

    def move_to_joint_target(
        self,
        target_rad: tuple[float, ...],
        *,
        timeout_s: float,
        settle: bool,
        progress_callback: ProgressCallback | None,
        waypoint_index: int,
    ) -> ControllerMoveResult: ...

    def set_gripper_width(
        self,
        *,
        mode: Literal["open", "close"],
        target_width_m: float,
        max_effort_n: float | None,
        timeout_s: float,
    ) -> BackendCallResult: ...

    def hold(
        self,
        *,
        duration_s: float | None,
        control_steps: int | None,
        timeout_s: float,
    ) -> BackendCallResult: ...

    def sample(self) -> Mapping[str, object]: ...

    def stop_and_hold(self, *, timeout_s: float) -> bool: ...


class LiberoControlPort:
    """Concrete narrow adapter over Cap-X's low-level LIBERO environment.

    ``env`` is the low-level object held by ``FrankaLiberoApi._env`` (or the
    code-execution environment's ``low_level_env``).  It must provide
    ``move_to_joints_blocking``, ``_set_gripper``, ``_step_once``, and either
    ``_current_arm_joint_positions`` or an explicit ``joint_reader``.
    """

    def __init__(
        self,
        env: object,
        *,
        max_gripper_width_m: float,
        joint_reader: Callable[[], Sequence[float]] | None = None,
        sample_provider: Callable[[], Mapping[str, object]] | None = None,
        sample_signal_names: Sequence[str] | None = None,
        sample_world_resource_bindings: Sequence[tuple[str, str]] | None = None,
        effort_setter: Callable[[float, float | None], None] | None = None,
        control_frequency_hz: float | None = None,
        joint_tolerance_rad: float = 0.01,
        max_joint_velocity_rad_s: float = 1.0,
        stall_patience_steps: int = 30,
        stall_min_progress_rad: float = 1e-4,
        final_settle_steps: int = 10,
        open_settle_steps: int = 40,
        close_settle_steps: int = 60,
        gripper_tolerance_m: float = 0.003,
    ) -> None:
        self.env = env
        self._move_blocking = _required_callable(env, "move_to_joints_blocking")
        self._set_gripper = _required_callable(env, "_set_gripper")
        self._step_once = _required_callable(env, "_step_once")
        discovered_reader = getattr(env, "_current_arm_joint_positions", None)
        self._joint_reader = joint_reader or (
            discovered_reader if callable(discovered_reader) else None
        )
        if self._joint_reader is None:
            raise CapXBackendConfigurationError(
                "LIBERO control env needs an explicit arm joint_reader"
            )
        discovered_frequency = getattr(env, "_control_freq", None)
        frequency = (
            float(control_frequency_hz)
            if control_frequency_hz is not None
            else float(discovered_frequency)
            if discovered_frequency is not None
            else math.nan
        )
        if not math.isfinite(frequency) or frequency <= 0:
            raise CapXBackendConfigurationError(
                "control_frequency_hz must be supplied and positive"
            )
        if not math.isfinite(max_gripper_width_m) or max_gripper_width_m <= 0:
            raise CapXBackendConfigurationError(
                "max_gripper_width_m must be explicit and positive"
            )
        if joint_tolerance_rad <= 0 or max_joint_velocity_rad_s <= 0:
            raise CapXBackendConfigurationError(
                "joint tolerance and velocity must be positive"
            )
        if min(
            stall_patience_steps,
            final_settle_steps,
            open_settle_steps,
            close_settle_steps,
        ) < 0:
            raise CapXBackendConfigurationError("controller step counts must be non-negative")
        self._control_frequency_hz = frequency
        self._max_gripper_width_m = float(max_gripper_width_m)
        if sample_provider is None:
            if sample_signal_names not in (None, (), []):
                raise CapXBackendConfigurationError(
                    "sample_signal_names require a sample_provider"
                )
            normalized_sample_names: tuple[str, ...] = ()
            if sample_world_resource_bindings not in (None, (), []):
                raise CapXBackendConfigurationError(
                    "sample bindings require a sample_provider"
                )
            normalized_sample_bindings: frozenset[tuple[str, str]] = frozenset()
        else:
            if sample_signal_names is None:
                raise CapXBackendConfigurationError(
                    "sample_provider must explicitly declare sample_signal_names"
                )
            normalized_sample_names = tuple(
                str(name).strip() for name in sample_signal_names
            )
            if not normalized_sample_names or not all(normalized_sample_names):
                raise CapXBackendConfigurationError(
                    "sample_signal_names must be a non-empty exact declaration"
                )
            if len(set(normalized_sample_names)) != len(normalized_sample_names):
                raise CapXBackendConfigurationError(
                    "sample_signal_names must be unique"
                )
            overlap = set(normalized_sample_names).intersection(
                _MONITOR_RESERVED_SIGNALS
            )
            if overlap:
                raise CapXBackendConfigurationError(
                    "sample_provider cannot overwrite reserved monitor signals: "
                    + ", ".join(sorted(overlap))
                )
            if sample_world_resource_bindings is None:
                raise CapXBackendConfigurationError(
                    "sample_provider must explicitly declare world/resource bindings"
                )
            normalized_bindings = tuple(
                (str(world_id).strip(), str(resource_id).strip())
                for world_id, resource_id in sample_world_resource_bindings
            )
            if (
                not normalized_bindings
                or not all(world and resource for world, resource in normalized_bindings)
                or len(set(normalized_bindings)) != len(normalized_bindings)
            ):
                raise CapXBackendConfigurationError(
                    "sample world/resource bindings must be unique and non-empty"
                )
            normalized_sample_bindings = frozenset(normalized_bindings)
        self._sample_provider = sample_provider
        self._sample_signal_names = normalized_sample_names
        self._sample_bindings = normalized_sample_bindings
        self._effort_setter = effort_setter
        self._joint_tolerance_rad = float(joint_tolerance_rad)
        self._max_joint_velocity_rad_s = float(max_joint_velocity_rad_s)
        self._stall_patience_steps = int(stall_patience_steps)
        self._stall_min_progress_rad = float(stall_min_progress_rad)
        self._final_settle_steps = int(final_settle_steps)
        self._open_settle_steps = int(open_settle_steps)
        self._close_settle_steps = int(close_settle_steps)
        self._gripper_tolerance_m = float(gripper_tolerance_m)
        self._tracking_step = getattr(env, "_tracking_step", None)

    @classmethod
    def from_code_execution_env(
        cls,
        env: object,
        *,
        api_name: str,
        max_gripper_width_m: float,
        **kwargs: object,
    ) -> LiberoControlPort:
        """Resolve one explicitly named Cap-X API and adapt its low-level env.

        Naming the API is mandatory so a multi-API code environment cannot silently
        select a different robot or world.
        """

        apis = getattr(env, "_apis", None)
        if not isinstance(apis, Mapping) or api_name not in apis:
            raise CapXBackendConfigurationError(
                f"Cap-X code environment has no configured API {api_name!r}"
            )
        api = apis[api_name]
        low_level = getattr(api, "_env", None)
        if low_level is None:
            raise CapXBackendConfigurationError(
                f"Cap-X API {api_name!r} does not expose its low-level control env"
            )
        return cls(
            low_level,
            max_gripper_width_m=max_gripper_width_m,
            **kwargs,
        )

    @property
    def control_frequency_hz(self) -> float:
        return self._control_frequency_hz

    @property
    def max_gripper_width_m(self) -> float:
        return self._max_gripper_width_m

    @property
    def monitor_signal_names(self) -> tuple[str, ...]:
        """Signals present in every phase/waypoint/control sample."""

        return tuple(sorted((*_CONTROL_ALWAYS_SIGNALS, *self._sample_signal_names)))

    @property
    def monitor_supported_hooks(self) -> tuple[MonitorTelemetryHook, ...]:
        hooks = [MonitorTelemetryHook.PHASE, MonitorTelemetryHook.WAYPOINT]
        if callable(self._tracking_step):
            hooks.append(MonitorTelemetryHook.CONTROL)
        return tuple(hooks)

    @property
    def cooperative_stop_guaranteed(self) -> bool:
        """A false callback is converted into a pinned physical hold target."""

        return True

    @property
    def monitor_bindings(self) -> frozenset[tuple[str, str]]:
        return self._sample_bindings

    @property
    def cooperative_granularity(self) -> Literal["control", "waypoint"]:
        """Finest safe interruption point available on the injected controller."""

        return "control" if callable(self._tracking_step) else "waypoint"

    def current_joint_positions(self) -> tuple[float, ...]:
        raw = np.asarray(self._joint_reader(), dtype=np.float64).reshape(-1)
        if raw.size == 0 or not np.isfinite(raw).all():
            raise CapXActionBackendError("controller returned an invalid arm joint state")
        return tuple(float(value) for value in raw)

    def move_to_joint_target(
        self,
        target_rad: tuple[float, ...],
        *,
        timeout_s: float,
        settle: bool,
        progress_callback: ProgressCallback | None,
        waypoint_index: int,
    ) -> ControllerMoveResult:
        target = np.asarray(target_rad, dtype=np.float64)
        current = np.asarray(self.current_joint_positions(), dtype=np.float64)
        if target.shape != current.shape:
            raise CapXActionBackendError(
                f"target width {target.size} differs from controller width {current.size}"
            )
        if not np.isfinite(target).all():
            raise CapXActionBackendError("joint target contains a non-finite value")
        if progress_callback is not None and callable(self._tracking_step):
            return self._move_with_control_progress(
                current,
                target,
                timeout_s=timeout_s,
                settle=settle,
                progress_callback=progress_callback,
                waypoint_index=waypoint_index,
            )

        started = time.monotonic()
        max_steps = max(1, math.floor(timeout_s * self.control_frequency_hz))
        status = self._move_blocking(
            target.copy(),
            tolerance=self._joint_tolerance_rad,
            max_steps=max_steps,
            settle_steps=self._final_settle_steps if settle else 0,
        )
        elapsed = time.monotonic() - started
        if not isinstance(status, Mapping):
            raise CapXActionBackendError(
                "move_to_joints_blocking must return an explicit status mapping"
            )
        timed_out = bool(status.get("timed_out", False)) or elapsed > timeout_s
        converged = bool(status.get("converged", False)) and not timed_out
        telemetry = {
            "waypoint_index": waypoint_index,
            "cooperative_granularity": self.cooperative_granularity,
            "elapsed_s": elapsed,
            "controller_status": _json_safe(status),
        }
        interrupted = False
        if progress_callback is not None and not timed_out:
            interrupted = not self._invoke_progress_callback(
                progress_callback,
                self._progress_sample(telemetry),
            )
            converged = converged and not interrupted
        self._pin_arm_hold_command(
            target if converged else np.asarray(self.current_joint_positions())
        )
        return ControllerMoveResult(
            converged=converged,
            timed_out=timed_out,
            interrupted=interrupted,
            stalled=bool(status.get("stalled", False)),
            telemetry=telemetry,
        )

    def _move_with_control_progress(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        timeout_s: float,
        settle: bool,
        progress_callback: ProgressCallback,
        waypoint_index: int,
    ) -> ControllerMoveResult:
        """Mirror LIBERO's min-jerk waypoint controller with a stop callback."""

        started = time.monotonic()
        deadline = started + timeout_s
        n_interp = _interpolation_step_budget(
            start,
            target,
            control_frequency_hz=self.control_frequency_hz,
            max_joint_velocity_rad_s=self._max_joint_velocity_rad_s,
        )
        adaptive_cap = max(180, n_interp + 60)
        timeout_cap = max(1, math.floor(timeout_s * self.control_frequency_hz))
        step_cap = min(adaptive_cap, timeout_cap)
        best_error = math.inf
        stall_counter = 0
        stalled = False
        steps = 0
        interrupted = False

        while steps < step_cap:
            current = np.asarray(self.current_joint_positions(), dtype=np.float64)
            error = float(np.linalg.norm(current - target))
            if error < self._joint_tolerance_rad and steps > 0:
                break
            if time.monotonic() >= deadline:
                break
            if error < best_error - self._stall_min_progress_rad:
                best_error = error
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter >= self._stall_patience_steps:
                    stalled = True
                    break
            alpha = _minimum_jerk((steps + 1) / n_interp)
            reference = start + alpha * (target - start)
            self._tracking_step(reference.copy())
            steps += 1
            current_after = np.asarray(self.current_joint_positions(), dtype=np.float64)
            error_after = float(np.linalg.norm(current_after - target))
            sample = self._progress_sample({
                "waypoint_index": waypoint_index,
                "control_step": steps,
                "cooperative_granularity": "control",
                "joint_positions_rad": [float(value) for value in current_after],
                "joint_target_rad": [float(value) for value in target],
                "joint_error_rad": error_after,
                "elapsed_s": time.monotonic() - started,
            })
            if not self._invoke_progress_callback(progress_callback, sample):
                interrupted = True
                break

        current = np.asarray(self.current_joint_positions(), dtype=np.float64)
        final_error = float(np.linalg.norm(current - target))
        if (
            not interrupted
            and not stalled
            and final_error < self._joint_tolerance_rad
            and settle
        ):
            for _ in range(self._final_settle_steps):
                if time.monotonic() >= deadline:
                    break
                self._tracking_step(target.copy())
                steps += 1
                current = np.asarray(self.current_joint_positions(), dtype=np.float64)
                final_error = float(np.linalg.norm(current - target))
                if not self._invoke_progress_callback(
                    progress_callback,
                    self._progress_sample({
                        "waypoint_index": waypoint_index,
                        "control_step": steps,
                        "cooperative_granularity": "control",
                        "settling": True,
                        "joint_positions_rad": [float(value) for value in current],
                        "joint_target_rad": [float(value) for value in target],
                        "joint_error_rad": final_error,
                        "elapsed_s": time.monotonic() - started,
                    }),
                ):
                    interrupted = True
                    break

        elapsed = time.monotonic() - started
        # Crossing the sealed deadline is indeterminate even when the controller
        # reports the target at the first observation after that deadline.  The
        # runtime must reconcile physical state instead of accepting a late success.
        timed_out = (
            not interrupted
            and not stalled
            and (
                elapsed >= timeout_s
                or (
                    final_error >= self._joint_tolerance_rad
                    and steps >= timeout_cap
                )
            )
        )
        converged = (
            final_error < self._joint_tolerance_rad
            and not interrupted
            and not timed_out
        )
        # LIBERO's ordinary blocking controller stores its target in
        # ``_current_joints``; later gripper/wait steps reuse that command.  On an
        # abort, pinning it to observed state is what makes interruption physical
        # instead of merely stopping this Python loop while the next step resumes
        # motion toward the old target.
        self._pin_arm_hold_command(
            target if converged else np.asarray(self.current_joint_positions())
        )
        return ControllerMoveResult(
            converged=converged,
            timed_out=timed_out,
            interrupted=interrupted,
            stalled=stalled,
            telemetry={
                "waypoint_index": waypoint_index,
                "cooperative_granularity": "control",
                "steps": steps,
                "step_cap": step_cap,
                "elapsed_s": elapsed,
                "final_error_rad": final_error,
                "stalled": stalled,
            },
        )

    def set_gripper_width(
        self,
        *,
        mode: Literal["open", "close"],
        target_width_m: float,
        max_effort_n: float | None,
        timeout_s: float,
    ) -> BackendCallResult:
        if target_width_m < 0 or target_width_m > self.max_gripper_width_m:
            raise CapXUnsupportedEffectError(
                "target gripper width is outside the explicitly configured range"
            )
        target_fraction = target_width_m / self.max_gripper_width_m
        if max_effort_n is not None and self._effort_setter is None:
            raise CapXUnsupportedEffectError(
                "this LIBERO controller cannot honor max_effort_n"
            )
        if self._effort_setter is not None:
            self._effort_setter(target_width_m, max_effort_n)
        else:
            self._set_gripper(target_fraction)

        required_steps = (
            self._open_settle_steps if mode == "open" else self._close_settle_steps
        )
        timeout_steps = max(0, math.floor(timeout_s * self.control_frequency_hz))
        executed_steps = min(required_steps, timeout_steps)
        started = time.monotonic()
        for _ in range(executed_steps):
            self._step_once()
        elapsed = time.monotonic() - started
        timed_out = executed_steps < required_steps or elapsed > timeout_s
        observed_width = self._observed_gripper_width()
        position_reached = (
            observed_width is not None
            and abs(observed_width - target_width_m) <= self._gripper_tolerance_m
        )
        # A close command may stop on a grasped object.  Completion means the exact
        # command was held for its bounded controller horizon, not that fingers passed
        # through the object to reach zero width.
        converged = not timed_out and (mode == "close" or position_reached)
        return BackendCallResult(
            converged=converged,
            timed_out=timed_out,
            telemetry={
                "mode": mode,
                "target_width_m": target_width_m,
                "target_fraction": target_fraction,
                "observed_width_m": observed_width,
                "position_reached": position_reached,
                "executed_control_steps": executed_steps,
                "required_control_steps": required_steps,
                "elapsed_s": elapsed,
            },
        )

    def hold(
        self,
        *,
        duration_s: float | None,
        control_steps: int | None,
        timeout_s: float,
    ) -> BackendCallResult:
        if (duration_s is None) == (control_steps is None):
            raise CapXUnsupportedEffectError(
                "hold requires exactly one duration or control-step bound"
            )
        requested_steps = (
            int(control_steps)
            if control_steps is not None
            else math.ceil(float(duration_s) * self.control_frequency_hz)
        )
        timeout_steps = max(0, math.floor(timeout_s * self.control_frequency_hz))
        executed_steps = min(requested_steps, timeout_steps)
        started = time.monotonic()
        for _ in range(executed_steps):
            self._step_once()
        elapsed = time.monotonic() - started
        timed_out = executed_steps < requested_steps or elapsed > timeout_s
        return BackendCallResult(
            converged=not timed_out,
            timed_out=timed_out,
            telemetry={
                "requested_control_steps": requested_steps,
                "executed_control_steps": executed_steps,
                "control_frequency_hz": self.control_frequency_hz,
                "elapsed_s": elapsed,
            },
        )

    def sample(self) -> Mapping[str, object]:
        sample: dict[str, object] = {
            "joint_positions_rad": list(self.current_joint_positions()),
            "cooperative_granularity": self.cooperative_granularity,
        }
        if self._sample_provider is not None:
            raw = self._sample_provider()
            if not isinstance(raw, Mapping):
                raise CapXActionBackendError(
                    "sample_provider must return a mapping"
                )
            actual_names = set(raw)
            if any(not isinstance(name, str) for name in actual_names):
                raise CapXActionBackendError(
                    "sample_provider returned a non-string key"
                )
            declared_names = set(self._sample_signal_names)
            if actual_names != declared_names:
                missing = declared_names.difference(actual_names)
                unknown = actual_names.difference(declared_names)
                details: list[str] = []
                if missing:
                    details.append("missing=" + repr(sorted(missing)))
                if unknown:
                    details.append("undeclared=" + repr(sorted(unknown)))
                raise CapXActionBackendError(
                    "sample_provider must return exactly sample_signal_names: "
                    + ", ".join(details)
                )
            normalized = _strict_json_safe(raw, path="sample_provider")
            assert isinstance(normalized, dict)
            sample.update(normalized)
        return sample

    def stop_and_hold(self, *, timeout_s: float) -> bool:
        """Replace the active target with observed joints and issue one hold step."""

        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise CapXUnsupportedEffectError("stop timeout must be finite and positive")
        started = time.monotonic()
        observed = np.asarray(self.current_joint_positions(), dtype=np.float64)
        self._pin_arm_hold_command(observed)
        if time.monotonic() >= started + timeout_s:
            return False
        self._step_once()
        return time.monotonic() - started <= timeout_s

    def _progress_sample(self, progress: Mapping[str, object]) -> Mapping[str, object]:
        try:
            sample = dict(self.sample())
        except BaseException:
            # A freshness/provider failure occurs after the most recent
            # discrete control step.  Pin the hold target before propagating so
            # no later controller call resumes toward the stale target.
            self._pin_arm_hold_command(
                np.asarray(self.current_joint_positions(), dtype=np.float64)
            )
            raise
        sample.update(progress)
        return sample

    def _invoke_progress_callback(
        self,
        callback: ProgressCallback,
        sample: Mapping[str, object],
    ) -> bool:
        try:
            return bool(callback(sample))
        except BaseException:
            self._pin_arm_hold_command(
                np.asarray(self.current_joint_positions(), dtype=np.float64)
            )
            raise

    def _pin_arm_hold_command(self, joints: np.ndarray) -> None:
        if hasattr(self.env, "_current_joints"):
            self.env._current_joints = np.asarray(  # type: ignore[attr-defined]
                joints, dtype=np.float64
            ).copy()

    def _observed_gripper_width(self) -> float | None:
        try:
            observation = self.env.get_observation()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - observation is telemetry, never authority
            return None
        if not isinstance(observation, Mapping):
            return None
        raw = observation.get("robot_joint_pos")
        if raw is None:
            return None
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        arm_width = len(self.current_joint_positions())
        if values.size <= arm_width or not math.isfinite(float(values[-1])):
            return None
        return float(np.clip(values[-1], 0.0, 1.0)) * self.max_gripper_width_m


class CapXSealedActionBackend:
    """ActionBackend executing only exact joint-path/gripper/wait effects."""

    def __init__(
        self,
        *,
        backend_id: str,
        control: CapXControlPort,
        snapshots: AdmissionSnapshotProvider,
        resource_roles: Mapping[str, ResourceRole | str],
        allowed_world_ids: Sequence[str],
    ) -> None:
        if not backend_id.strip():
            raise CapXBackendConfigurationError("backend_id must not be empty")
        roles = {
            str(resource_id): ResourceRole(role)
            for resource_id, role in resource_roles.items()
        }
        if not roles:
            raise CapXBackendConfigurationError("resource_roles must not be empty")
        worlds = frozenset(str(world_id).strip() for world_id in allowed_world_ids)
        if not worlds or not all(worlds):
            raise CapXBackendConfigurationError("allowed_world_ids must not be empty")
        try:
            monitor_signal_names = tuple(control.monitor_signal_names)
            monitor_supported_hooks = tuple(
                MonitorTelemetryHook(value)
                for value in control.monitor_supported_hooks
            )
            cooperative_stop_guaranteed = bool(
                control.cooperative_stop_guaranteed
            )
            monitor_bindings = frozenset(
                (str(world).strip(), str(resource).strip())
                for world, resource in control.monitor_bindings
            )
        except Exception as exc:
            raise CapXBackendConfigurationError(
                "control port must declare monitor signals and hooks"
            ) from exc
        if (
            not monitor_signal_names
            or len(set(monitor_signal_names)) != len(monitor_signal_names)
            or not all(isinstance(name, str) and name.strip() for name in monitor_signal_names)
        ):
            raise CapXBackendConfigurationError(
                "control monitor_signal_names must be a unique non-empty exact set"
            )
        if (
            not monitor_supported_hooks
            or len(set(monitor_supported_hooks)) != len(monitor_supported_hooks)
        ):
            raise CapXBackendConfigurationError(
                "control monitor_supported_hooks must be a unique non-empty set"
            )
        configured_bindings = {
            (world, resource)
            for world in worlds
            for resource in roles
        }
        if not monitor_bindings.issubset(configured_bindings):
            raise CapXBackendConfigurationError(
                "control monitor binding is not present in allowed worlds/resources"
            )
        self._descriptor = BackendDescriptor(
            backend_id=backend_id,
            motion_interface=BackendMotionInterface.EXACT_JOINT_PATH,
            # Libero/MuJoCo control surfaces are not assumed safe for an
            # out-of-band stop while a watchdog worker is still inside env.step.
            watchdog_stop_thread_safe=False,
        )
        self.control = control
        self.snapshots = snapshots
        self.resource_roles = roles
        self.allowed_world_ids = worlds
        self._monitor_signal_names = monitor_signal_names
        self._monitor_supported_hooks = monitor_supported_hooks
        self._cooperative_stop_guaranteed = cooperative_stop_guaranteed
        self._monitor_bindings = monitor_bindings

    @property
    def descriptor(self) -> BackendDescriptor:
        # Return a validated copy so callers cannot rely on mutable adapter state.
        return BackendDescriptor.model_validate(
            self._descriptor.model_dump(mode="python")
        )

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        self._assert_binding(world_id, resource_id)
        snapshot = self.snapshots.snapshot(world_id, resource_id)
        validated = AdmissionSnapshot.model_validate(
            snapshot.model_dump(mode="python")
        )
        if validated.world_id != world_id or validated.resource_id != resource_id:
            raise CapXBackendConfigurationError(
                "snapshot provider returned a different world/resource binding"
            )
        return validated

    def monitor_telemetry_capabilities(
        self,
        *,
        world_id: str,
        resource_id: str,
    ) -> MonitorTelemetryCapabilities:
        """Declare the exact sample contract for one bound Cap-X resource."""

        self._assert_binding(world_id, resource_id)
        if self._monitor_bindings and (
            world_id,
            resource_id,
        ) not in self._monitor_bindings:
            raise CapXBackendConfigurationError(
                "custom monitor telemetry is bound to another world/resource"
            )
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=self._monitor_signal_names,
            supported_hooks=self._monitor_supported_hooks,
            cooperative_stop_guaranteed=self._cooperative_stop_guaranteed,
        )

    def execute_joint_path(
        self,
        *,
        world_id: str,
        resource_id: str,
        joint_names: tuple[str, ...],
        positions_rad: tuple[tuple[float, ...], ...],
        mode: str,
        subsample: int,
        timeout_s: float,
    ) -> BackendCallResult:
        return self._execute_joint_path(
            world_id=world_id,
            resource_id=resource_id,
            joint_names=joint_names,
            positions_rad=positions_rad,
            mode=mode,
            subsample=subsample,
            timeout_s=timeout_s,
            progress_callback=None,
        )

    def execute_joint_path_cooperative(
        self,
        *,
        progress_callback: ProgressCallback,
        world_id: str,
        resource_id: str,
        joint_names: tuple[str, ...],
        positions_rad: tuple[tuple[float, ...], ...],
        mode: str,
        subsample: int,
        timeout_s: float,
    ) -> BackendCallResult:
        if not callable(progress_callback):
            raise CapXUnsupportedEffectError("progress_callback must be callable")
        return self._execute_joint_path(
            world_id=world_id,
            resource_id=resource_id,
            joint_names=joint_names,
            positions_rad=positions_rad,
            mode=mode,
            subsample=subsample,
            timeout_s=timeout_s,
            progress_callback=progress_callback,
        )

    def _execute_joint_path(
        self,
        *,
        world_id: str,
        resource_id: str,
        joint_names: tuple[str, ...],
        positions_rad: tuple[tuple[float, ...], ...],
        mode: str,
        subsample: int,
        timeout_s: float,
        progress_callback: ProgressCallback | None,
    ) -> BackendCallResult:
        self._assert_binding(world_id, resource_id, ResourceRole.ARM)
        if mode != "blocking_waypoint":
            raise CapXUnsupportedEffectError(f"unsupported exact path mode {mode!r}")
        if subsample < 1:
            raise CapXUnsupportedEffectError("subsample must be at least one")
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise CapXUnsupportedEffectError("timeout_s must be finite and positive")
        current_snapshot = self.snapshot(world_id, resource_id)
        if current_snapshot.controller_state not in {
            ControllerState.READY,
            ControllerState.QUIESCENT,
        }:
            raise CapXUnsupportedEffectError(
                f"controller is not executable: {current_snapshot.controller_state.value}"
            )
        if joint_names != current_snapshot.joint_names:
            raise CapXUnsupportedEffectError(
                "joint order does not match the authoritative controller snapshot"
            )
        width = len(joint_names)
        if not positions_rad:
            raise CapXUnsupportedEffectError("exact joint path must not be empty")
        if any(
            len(waypoint) != width
            or not all(math.isfinite(value) for value in waypoint)
            for waypoint in positions_rad
        ):
            raise CapXUnsupportedEffectError("joint path shape/value validation failed")

        indices = list(range(0, len(positions_rad), subsample))
        if indices[-1] != len(positions_rad) - 1:
            indices.append(len(positions_rad) - 1)
        started = time.monotonic()
        statuses: list[dict[str, JsonValue]] = []
        for selected_position, waypoint_index in enumerate(indices):
            elapsed = time.monotonic() - started
            remaining = timeout_s - elapsed
            if remaining <= 0:
                return self._path_result(
                    statuses,
                    converged=False,
                    timed_out=True,
                    interrupted=False,
                    elapsed_s=elapsed,
                    selected_indices=indices,
                )
            move = self.control.move_to_joint_target(
                positions_rad[waypoint_index],
                timeout_s=remaining,
                settle=selected_position == len(indices) - 1,
                progress_callback=progress_callback,
                waypoint_index=waypoint_index,
            )
            statuses.append({
                "waypoint_index": waypoint_index,
                "converged": move.converged,
                "timed_out": move.timed_out,
                "interrupted": move.interrupted,
                "stalled": move.stalled,
                "telemetry": _json_safe(move.telemetry or {}),
            })
            if move.timed_out or move.interrupted or not move.converged:
                return self._path_result(
                    statuses,
                    converged=False,
                    timed_out=move.timed_out,
                    interrupted=move.interrupted,
                    elapsed_s=time.monotonic() - started,
                    selected_indices=indices,
                )
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            return self._path_result(
                statuses,
                converged=False,
                timed_out=True,
                interrupted=False,
                elapsed_s=elapsed,
                selected_indices=indices,
            )
        return self._path_result(
            statuses,
            converged=True,
            timed_out=False,
            interrupted=False,
            elapsed_s=elapsed,
            selected_indices=indices,
        )

    @staticmethod
    def _path_result(
        statuses: list[dict[str, JsonValue]],
        *,
        converged: bool,
        timed_out: bool,
        interrupted: bool,
        elapsed_s: float,
        selected_indices: list[int],
    ) -> BackendCallResult:
        return BackendCallResult(
            converged=converged,
            timed_out=timed_out,
            interrupted=interrupted,
            telemetry={
                "selected_waypoint_indices": selected_indices,
                "completed_waypoints": len(statuses),
                "waypoint_statuses": statuses,
                "elapsed_s": elapsed_s,
                "timeout_semantics": (
                    "indeterminate_after_timeout_if_any_control_step_was_issued"
                ),
            },
        )

    def set_gripper(
        self,
        *,
        world_id: str,
        resource_id: str,
        mode: str,
        target_width_m: float,
        max_effort_n: float | None,
        timeout_s: float,
    ) -> BackendCallResult:
        self._assert_binding(world_id, resource_id, ResourceRole.GRIPPER)
        current_snapshot = self.snapshot(world_id, resource_id)
        if current_snapshot.controller_state not in {
            ControllerState.READY,
            ControllerState.QUIESCENT,
        }:
            raise CapXUnsupportedEffectError(
                f"controller is not executable: {current_snapshot.controller_state.value}"
            )
        if mode not in {"open", "close"}:
            raise CapXUnsupportedEffectError(f"unsupported gripper mode {mode!r}")
        return self.control.set_gripper_width(
            mode=mode,
            target_width_m=target_width_m,
            max_effort_n=max_effort_n,
            timeout_s=timeout_s,
        )

    def wait(
        self,
        *,
        world_id: str,
        resource_id: str,
        duration_s: float | None,
        control_steps: int | None,
        hold_command: str,
        timeout_s: float,
    ) -> BackendCallResult:
        self._assert_binding(world_id, resource_id, ResourceRole.CONTROLLER)
        current_snapshot = self.snapshot(world_id, resource_id)
        if current_snapshot.controller_state not in {
            ControllerState.READY,
            ControllerState.QUIESCENT,
        }:
            raise CapXUnsupportedEffectError(
                f"controller is not executable: {current_snapshot.controller_state.value}"
            )
        if hold_command != "hold_current":
            raise CapXUnsupportedEffectError(f"unsupported hold command {hold_command!r}")
        return self.control.hold(
            duration_s=duration_s,
            control_steps=control_steps,
            timeout_s=timeout_s,
        )

    def monitor_sample(
        self,
        *,
        world_id: str,
        resource_id: str,
        phase: str,
        sequence: int,
    ) -> Mapping[str, object]:
        self._assert_binding(world_id, resource_id)
        self.monitor_telemetry_capabilities(
            world_id=world_id,
            resource_id=resource_id,
        )
        return {
            **self.control.sample(),
            "phase": phase,
            "sequence": sequence,
            "world_id": world_id,
            "resource_id": resource_id,
        }

    def stop_and_wait_quiescent(
        self,
        *,
        world_id: str,
        resource_id: str,
        timeout_s: float,
    ) -> AdmissionSnapshot:
        """Request a controller hold, then require authoritative quiescence proof."""

        self._assert_binding(world_id, resource_id)
        stop = getattr(self.control, "stop_and_hold", None)
        if not callable(stop):
            raise CapXUnsupportedEffectError(
                "control port does not expose a watchdog stop primitive"
            )
        started = time.monotonic()
        if not bool(stop(timeout_s=timeout_s)):
            raise CapXActionBackendError("controller did not acknowledge the stop request")
        deadline = started + timeout_s
        last = self.snapshot(world_id, resource_id)
        while last.controller_state is not ControllerState.QUIESCENT:
            if time.monotonic() >= deadline:
                raise CapXActionBackendError(
                    "controller stop was not followed by a quiescent snapshot"
                )
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            last = self.snapshot(world_id, resource_id)
        return last

    def _assert_binding(
        self,
        world_id: str,
        resource_id: str,
        expected_role: ResourceRole | None = None,
    ) -> None:
        if world_id not in self.allowed_world_ids:
            raise CapXUnsupportedEffectError(f"world {world_id!r} is not bound to backend")
        role = self.resource_roles.get(resource_id)
        if role is None:
            raise CapXUnsupportedEffectError(
                f"resource {resource_id!r} is not bound to backend"
            )
        if expected_role is not None and role is not expected_role:
            raise CapXUnsupportedEffectError(
                f"resource {resource_id!r} is {role.value}, expected {expected_role.value}"
            )


@dataclass(frozen=True)
class FeasibilityProbeOutcome:
    status: FeasibilityStatus
    evidence_refs: tuple[str, ...] = ()


FeasibilityProbeResult: TypeAlias = (  # noqa: UP040
    FeasibilityProbeOutcome | FeasibilityStatus | str | bool
)
MotionCollisionProbe: TypeAlias = Callable[  # noqa: UP040
    [MotionPlan, AdmissionSnapshot], FeasibilityProbeResult
]


class CapXFeasibilityChecker:
    """Fail-closed checker bound to the exact spec and admission snapshot.

    Joint limits and robot-model identity are checked locally from explicit pins.
    Cap-X/LIBERO does not expose a universal read-only collision checker, so motion
    collision certification is an injected callback (typically CuRobo against the
    collision-world digest in ``snapshot``).  Omitting any required motion evidence
    yields ``UNKNOWN`` and therefore a non-admissible certificate.
    """

    def __init__(
        self,
        *,
        checker_id: str,
        backend: CapXSealedActionBackend,
        robot_model_digest: str | None,
        joint_limits_rad: Mapping[str, tuple[float, float]] | None,
        collision_probe: MotionCollisionProbe | None,
        supports_gripper_effort: bool = False,
    ) -> None:
        if not checker_id.strip():
            raise CapXBackendConfigurationError("checker_id must not be empty")
        self.checker_id = checker_id
        self.backend = backend
        self.robot_model_digest = robot_model_digest
        self.joint_limits_rad = dict(joint_limits_rad or {})
        self.collision_probe = collision_probe
        self.supports_gripper_effort = bool(supports_gripper_effort)

    def certify(
        self,
        spec: ActionSpec,
        snapshot: AdmissionSnapshot,
    ) -> FeasibilityCertificate:
        sealed = validate_action_spec(spec)
        current = AdmissionSnapshot.model_validate(snapshot.model_dump(mode="python"))
        checks: dict[str, FeasibilityStatus] = {
            "exact_joint_path_interface": FeasibilityStatus.PASS,
            "controller_admissible": (
                FeasibilityStatus.PASS
                if current.controller_state
                in {ControllerState.READY, ControllerState.QUIESCENT}
                else FeasibilityStatus.FAIL
            ),
            "resource_binding": self._resource_binding_status(sealed),
        }
        evidence_refs: list[str] = []
        if isinstance(sealed, MotionPlan):
            checks.update(self._motion_checks(sealed, current, evidence_refs))
        elif isinstance(sealed, GripperCommand):
            checks.update(self._gripper_checks(sealed))
        elif isinstance(sealed, WaitSpec):
            checks.update(self._wait_checks(sealed))
        else:  # pragma: no cover - validate_action_spec closes the union
            raise TypeError(f"unsupported action spec {type(sealed)!r}")
        return build_feasibility_certificate(
            spec=sealed,
            snapshot=current,
            checker_id=self.checker_id,
            checks=checks,
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        )

    def _motion_checks(
        self,
        spec: MotionPlan,
        snapshot: AdmissionSnapshot,
        evidence_refs: list[str],
    ) -> dict[str, FeasibilityStatus]:
        limits_status = FeasibilityStatus.PASS
        if not self.joint_limits_rad or any(
            name not in self.joint_limits_rad for name in spec.motion.joint_names
        ):
            limits_status = FeasibilityStatus.UNKNOWN
        else:
            for waypoint in spec.motion.positions_rad:
                if any(
                    value < self.joint_limits_rad[name][0]
                    or value > self.joint_limits_rad[name][1]
                    for name, value in zip(
                        spec.motion.joint_names, waypoint, strict=True
                    )
                ):
                    limits_status = FeasibilityStatus.FAIL
                    break

        if self.robot_model_digest is None:
            model_status = FeasibilityStatus.UNKNOWN
        else:
            model_status = (
                FeasibilityStatus.PASS
                if spec.robot_model_digest == self.robot_model_digest
                else FeasibilityStatus.FAIL
            )
        collision_status = FeasibilityStatus.UNKNOWN
        if self.collision_probe is not None:
            outcome = _normalize_probe_outcome(self.collision_probe(spec, snapshot))
            collision_status = outcome.status
            evidence_refs.extend(outcome.evidence_refs)
        return {
            "joint_order": (
                FeasibilityStatus.PASS
                if spec.motion.joint_names == snapshot.joint_names
                else FeasibilityStatus.FAIL
            ),
            "joint_limits": limits_status,
            "robot_model": model_status,
            "collision": collision_status,
        }

    def _gripper_checks(self, spec: GripperCommand) -> dict[str, FeasibilityStatus]:
        target_status = (
            FeasibilityStatus.PASS
            if 0 <= spec.target_width_m <= self.backend.control.max_gripper_width_m
            else FeasibilityStatus.FAIL
        )
        effort_status = (
            FeasibilityStatus.NOT_APPLICABLE
            if spec.max_effort_n is None
            else FeasibilityStatus.PASS
            if self.supports_gripper_effort
            else FeasibilityStatus.FAIL
        )
        return {
            "gripper_target_range": target_status,
            "gripper_effort_support": effort_status,
        }

    def _wait_checks(self, spec: WaitSpec) -> dict[str, FeasibilityStatus]:
        if spec.control_steps is not None:
            required_s = spec.control_steps / self.backend.control.control_frequency_hz
        else:
            required_s = float(spec.duration_s)
        return {
            "wait_timeout_bound": (
                FeasibilityStatus.PASS
                if required_s <= spec.timeout_s
                else FeasibilityStatus.FAIL
            )
        }

    def _resource_binding_status(self, spec: ActionSpec) -> FeasibilityStatus:
        expected = (
            ResourceRole.ARM
            if isinstance(spec, MotionPlan)
            else ResourceRole.GRIPPER
            if isinstance(spec, GripperCommand)
            else ResourceRole.CONTROLLER
        )
        role = self.backend.resource_roles.get(spec.resource_id)
        return (
            FeasibilityStatus.PASS
            if spec.world_id in self.backend.allowed_world_ids and role is expected
            else FeasibilityStatus.FAIL
        )


def _normalize_probe_outcome(value: FeasibilityProbeResult) -> FeasibilityProbeOutcome:
    if isinstance(value, FeasibilityProbeOutcome):
        return value
    if isinstance(value, bool):
        return FeasibilityProbeOutcome(
            FeasibilityStatus.PASS if value else FeasibilityStatus.FAIL
        )
    return FeasibilityProbeOutcome(FeasibilityStatus(value))


def _required_callable(owner: object, name: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise CapXBackendConfigurationError(
            f"Cap-X control env must provide callable {name}"
        )
    return value


def _minimum_jerk(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _interpolation_step_budget(
    start: np.ndarray,
    target: np.ndarray,
    *,
    control_frequency_hz: float,
    max_joint_velocity_rad_s: float,
) -> int:
    distance = float(np.max(np.abs(target - start)))
    steps = math.ceil(
        distance / max(max_joint_velocity_rad_s, 1e-6)
        * control_frequency_hz
        * 1.5
    )
    return max(10, steps + 10)


def _json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _strict_json_safe(value: object, *, path: str) -> JsonValue:
    """Validate monitor telemetry without repr-based information loss."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapXActionBackendError(f"{path} contains NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _strict_json_safe(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _strict_json_safe(value.tolist(), path=path)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CapXActionBackendError(f"{path} contains a non-string key")
            normalized[key] = _strict_json_safe(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CapXActionBackendError(
        f"{path} contains unsupported type {type(value).__name__}"
    )


__all__ = [
    "AdmissionSnapshotProvider",
    "CallbackAdmissionSnapshotProvider",
    "CapXActionBackendError",
    "CapXBackendConfigurationError",
    "CapXControlPort",
    "CapXFeasibilityChecker",
    "CapXSealedActionBackend",
    "CapXUnsupportedEffectError",
    "ControllerMoveResult",
    "FeasibilityProbeOutcome",
    "JointState",
    "LiberoControlPort",
    "ResourceRole",
    "SnapshotRevisions",
]
