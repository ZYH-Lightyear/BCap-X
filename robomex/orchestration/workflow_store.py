"""Immutable workflow-open descriptors for deterministic Episode recovery."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.data import ResolvedArtifactRef
from robomex.elastic.graph_patch import ComposableFrontier
from robomex.orchestration.intent import SubgoalIntent


class WorkflowDescriptorError(RuntimeError):
    """A workflow identity was rebound or its durable descriptor is corrupt."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ArtifactRefRecord(_StrictModel):
    artifact_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @classmethod
    def from_ref(cls, ref: ResolvedArtifactRef) -> ArtifactRefRecord:
        return cls(**ref.to_mapping())

    def to_ref(self) -> ResolvedArtifactRef:
        return ResolvedArtifactRef(self.artifact_id, self.content_digest)


class WorkflowDescriptor(_StrictModel):
    schema_version: Literal["robomex.workflow_descriptor.v1"] = (
        "robomex.workflow_descriptor.v1"
    )
    episode_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    intent: SubgoalIntent
    initial_graph_id: str = Field(min_length=1)
    initial_graph_revision: int = Field(ge=1)
    initial_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_refs: dict[str, ArtifactRefRecord] = Field(default_factory=dict)
    frontier: ComposableFrontier | None = None
    manager_session_id: str | None = Field(default=None, min_length=1)
    content_digest: str = ""

    @model_validator(mode="after")
    def _seal(self) -> WorkflowDescriptor:
        if any(not key.strip() for key in self.external_refs):
            raise ValueError("workflow external-ref names must not be empty")
        if self.frontier is not None and (
            self.frontier.graph_id != self.initial_graph_id
            or self.frontier.revision != self.initial_graph_revision
            or self.frontier.graph_digest != self.initial_graph_digest
        ):
            raise ValueError("workflow frontier is not bound to the initial graph")
        payload = self.model_dump(mode="json", exclude={"content_digest"})
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if self.content_digest and self.content_digest != expected:
            raise ValueError("workflow descriptor content digest mismatch")
        if not self.content_digest:
            object.__setattr__(self, "content_digest", expected)
        return self


class WorkflowDescriptorStore:
    """Create-once workflow descriptors; scheduler state may evolve separately."""

    def __init__(self, root: str | Path, *, episode_id: str) -> None:
        self.root = Path(root)
        self.episode_id = episode_id
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, workflow_id: str) -> Path:
        if not workflow_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in workflow_id
        ):
            raise WorkflowDescriptorError("workflow_id is not path safe")
        return self.root / f"{workflow_id}.workflow.v1.json"

    def load(self, workflow_id: str) -> WorkflowDescriptor | None:
        path = self.path_for(workflow_id)
        if not path.exists():
            return None
        try:
            descriptor = WorkflowDescriptor.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise WorkflowDescriptorError(
                f"invalid workflow descriptor {workflow_id!r}"
            ) from exc
        if descriptor.episode_id != self.episode_id:
            raise WorkflowDescriptorError("workflow descriptor belongs to another episode")
        return descriptor

    def bind(self, descriptor: WorkflowDescriptor) -> WorkflowDescriptor:
        validated = WorkflowDescriptor.model_validate(
            descriptor.model_dump(mode="python")
        )
        if validated.episode_id != self.episode_id:
            raise WorkflowDescriptorError("cannot bind a foreign episode descriptor")
        path = self.path_for(validated.workflow_id)
        encoded = (
            json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        temporary = self.root / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor_fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor_fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            try:
                # A hard-link is an atomic create-if-absent operation.  Readers
                # can therefore never observe a partially written descriptor.
                os.link(temporary, path)
            except FileExistsError:
                existing = self.load(validated.workflow_id)
                if existing != validated:
                    raise WorkflowDescriptorError(
                        f"workflow {validated.workflow_id!r} was rebound"
                    ) from None
                return existing
            directory_fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        return validated


__all__ = [
    "ArtifactRefRecord",
    "WorkflowDescriptor",
    "WorkflowDescriptorError",
    "WorkflowDescriptorStore",
]
