from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCompilesProbe(ProbeAdapter):
    """
    Label: whether the current code state is syntactically valid Python (compiles without error).
    Changes: at each EditEvent where ast.parse or compile() succeeds/fails.
    None: before the first EditEvent (no code state to evaluate yet).
    Requires: agent mode with edit_history populated; ast.parse on each edit's code field.
    """

    name = "currently_compiles"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        if not ctx.edit_history:
            raise NotImplementedError("requires agent-mode task with edit_history")
        # TODO: carry-forward the label from the most recent EditEvent at or before each step.
        # For each step t, find the last EditEvent with step_idx <= t, try ast.parse(edit.code),
        # return True if it succeeds, False otherwise. Return None before first edit.
        return [None] * ctx.n_captured_steps
