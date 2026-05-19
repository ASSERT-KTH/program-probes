from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EditEvent:
    step_idx: int
    code: str
    test_results: dict | None
    compiles: bool | None = None


@dataclass
class TrajectoryContext:
    sample: dict
    generated_text: str
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
        Dynamic (is_dynamic=True): return a list of length len(ctx.edit_history).
          Each entry is the label for the corresponding EditEvent in edit_history.
          None values are masked out during probe training.
        """
        ...
