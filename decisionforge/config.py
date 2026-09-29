from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Hard limits that keep the agent loop bounded and auditable."""

    max_tool_calls: int = 12
    max_followups: int = 6
    max_repair_rounds: int = 1
    max_rows: int = 200
