from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCompilesProbe(ProbeAdapter):
    """Label: whether the code at each edit compiles without errors.

    Returns one label per EditEvent from the ``compiles`` field.
    A ``compiles`` of *None* produces a *None* label.
    """

    name = "currently_compiles"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        return [edit.compiles for edit in ctx.edit_history]
