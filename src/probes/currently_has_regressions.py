from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyHasRegressionsProbe(ProbeAdapter):
    """
    Label: whether the current edit introduced regressions (tests that passed before now fail).
    Changes: at each EditEvent where test_results["failed"] contains tests that were in
             the previous edit's test_results["passed"].
    None: before the first EditEvent (no baseline to compare against).
    Requires: agent mode with edit_history populated; test_results must include "passed"/"failed"
              lists per edit to compare across consecutive edits.
    """

    name = "currently_has_regressions"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        if not ctx.edit_history:
            raise NotImplementedError("requires agent-mode task with edit_history")
        # TODO: carry-forward the label from the most recent EditEvent at or before each step.
        # For each step t, find the last EditEvent with step_idx <= t. Check if any test in
        # test_results["failed"] also appeared in the prior EditEvent's test_results["passed"].
        # Return True if regressions exist, False otherwise. Return None before first edit.
        return [None] * ctx.n_captured_steps
