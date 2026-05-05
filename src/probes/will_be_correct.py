from src.probes.base import ProbeAdapter, TrajectoryContext
from src.tasks.base import TaskAdapter


class WillBeCorrectProbe(ProbeAdapter):
    name = "will_be_correct"
    is_dynamic = False

    def __init__(self, task_adapter: TaskAdapter):
        self._task_adapter = task_adapter

    def compute_label(self, ctx: TrajectoryContext) -> bool | None:
        return self._task_adapter.check_correct(ctx.generated_text, ctx.sample)
