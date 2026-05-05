from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyReducesFailingProbe(ProbeAdapter):
    """
    Label: whether the current edit reduced the number of failing tests compared to the previous edit.
    Changes: at each EditEvent where len(test_results["failed"]) < len(previous_edit["failed"]).
    None: before the first EditEvent (no baseline to compare against).
    Requires: agent mode with edit_history populated; test_results must include "failed" lists per edit.
    """

    name = "currently_reduces_failing"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        if not ctx.edit_history:
            raise NotImplementedError("requires agent-mode task with edit_history")
        # TODO: carry-forward the label from the most recent EditEvent at or before each step.
        # For each step t, find the last EditEvent with step_idx <= t. Compare its
        # test_results["failed"] count with the prior EditEvent's count. Return True if reduced,
        # False otherwise. Return None before first edit (no baseline).
        return [None] * ctx.n_captured_steps
