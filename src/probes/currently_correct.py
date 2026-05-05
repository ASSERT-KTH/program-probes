from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCorrectProbe(ProbeAdapter):
    """
    Label: whether the current code state passes all tests.
    Changes: at each EditEvent where test_results shows all tests passing.
    None: before the first EditEvent (no code state to evaluate yet).
    Requires: agent mode with edit_history populated; evalplus test harness per edit.
    """

    name = "currently_correct"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        if not ctx.edit_history:
            raise NotImplementedError("requires agent-mode task with edit_history")
        # TODO: carry-forward the label from the most recent EditEvent at or before each step.
        # For each step t, find the last EditEvent with step_idx <= t, check if
        # test_results["failed"] is empty, and return True/False. Return None before first edit.
        return [None] * ctx.n_captured_steps
