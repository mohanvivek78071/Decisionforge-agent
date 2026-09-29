from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    """The only surface the agents depend on: task name + JSON payload -> typed object.

    Tasks: "plan", "next_step", "narrate". Swap providers without touching agent code.
    """

    name: str

    def structured(self, task: str, payload: dict, model: type[T]) -> T: ...
