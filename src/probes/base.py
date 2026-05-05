from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EditEvent:
    step_idx: int
    code: str
    test_results: dict


@dataclass
class TrajectoryContext:
    sample: dict
    generated_text: str
    n_captured_steps: int
    edit_history: list[EditEvent] = field(default_factory=list)


class ProbeAdapter(ABC):
    name: str
    is_dynamic: bool

    @abstractmethod
    def compute_label(
        self, ctx: TrajectoryContext
    ) -> "bool | None | list[bool | None]":
        """
        Static (is_dynamic=False): return a single bool or None.
        Dynamic (is_dynamic=True): return a list of length ctx.n_captured_steps.
          Each entry is the carry-forwarded label from the most recent EditEvent
          at or before that step, or None if no edit has occurred yet.
          None values are masked out during probe training.
        """
        ...
