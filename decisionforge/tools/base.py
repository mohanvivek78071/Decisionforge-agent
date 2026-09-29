"""Typed tool registry. Every tool has a pydantic arg schema, so the same
definition powers validation, LLM tool descriptions and error messages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, BaseModel], tuple[str, dict[str, Any]]]


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, args_model: type[BaseModel]):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, args_model, fn)
        return fn

    return deco


def catalogue() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "args_schema": t.args_model.model_json_schema()}
        for t in REGISTRY.values()
    ]


def run_tool(store, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate args, run the tool. Returns (validated_args, summary, data)."""
    t = REGISTRY.get(name)
    if t is None:
        raise ToolError(f"unknown tool '{name}'. available: {sorted(REGISTRY)}")
    try:
        parsed = t.args_model(**args)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        raise ToolError(f"invalid arguments for {name}: {msg}") from e
    summary, data = t.fn(store, parsed)
    return parsed.model_dump(), summary, data
