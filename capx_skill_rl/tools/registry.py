"""The complete and intentionally small model-facing tool registry."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from capx_skill_rl.context import EnvContext

ToolHandler = Callable[[EnvContext, dict[str, Any]], dict[str, Any]]
Validator = Callable[[Mapping[str, Any]], dict[str, Any]]


class ToolInputError(ValueError):
    """A policy-visible tool schema violation."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    validate: Validator
    handler: ToolHandler
    physical: bool = False

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("tool names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self._specs.values()]

    def prepare(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> tuple[ToolSpec, dict[str, Any]]:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolInputError(
                f"unknown tool {name!r}; available tools: {list(self._specs)}"
            )
        return spec, spec.validate(arguments)


def object_schema(
    properties: dict[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def exact_keys(
    arguments: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise ToolInputError("arguments must be an object")
    allowed = required if allowed is None else allowed
    keys = set(arguments)
    missing = required - keys
    extra = keys - allowed
    if missing:
        raise ToolInputError(f"missing argument(s): {sorted(missing)}")
    if extra:
        raise ToolInputError(f"unexpected argument(s): {sorted(extra)}")
    return dict(arguments)


def string_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} must be a non-empty string")
    return value.strip()


def number_array(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ToolInputError(f"{name} must be an array of exactly {length} numbers")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ToolInputError(f"{name} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ToolInputError(f"{name} must contain only finite numbers")
        output.append(number)
    return output


def validate_empty(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return exact_keys(arguments, required=set())


def validate_query(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = exact_keys(arguments, required={"query"})
    return {"query": string_value(values["query"], "query")}


def validate_mask_id(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = exact_keys(arguments, required={"mask_id"})
    return {"mask_id": string_value(values["mask_id"], "mask_id")}


def validate_sam3(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = exact_keys(
        arguments,
        required=set(),
        allowed={"text", "bbox", "point"},
    )
    if len(values) != 1:
        raise ToolInputError("sam3 requires exactly one of text, bbox, or point")
    if "text" in values:
        return {"text": string_value(values["text"], "text")}
    if "bbox" in values:
        bbox = number_array(values["bbox"], "bbox", 4)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ToolInputError("bbox must satisfy x2 > x1 and y2 > y1")
        return {"bbox": bbox}
    return {"point": number_array(values["point"], "point", 2)}


def validate_pose(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = exact_keys(arguments, required={"position", "quaternion"})
    quaternion = number_array(values["quaternion"], "quaternion", 4)
    if math.sqrt(sum(value * value for value in quaternion)) < 1e-8:
        raise ToolInputError("quaternion must be non-zero")
    return {
        "position": number_array(values["position"], "position", 3),
        "quaternion": quaternion,
    }


def validate_joints(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = exact_keys(arguments, required={"joints"})
    return {"joints": number_array(values["joints"], "joints", 7)}


__all__ = [
    "ToolInputError",
    "ToolRegistry",
    "ToolSpec",
    "exact_keys",
    "number_array",
    "object_schema",
    "validate_empty",
    "validate_joints",
    "validate_mask_id",
    "validate_pose",
    "validate_query",
    "validate_sam3",
]
