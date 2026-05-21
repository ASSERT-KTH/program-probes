from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCompilesSwEProbe(ProbeAdapter):
    """Dynamic probe: True at each edit if the codebase compiled at that step."""

    name = "currently_compiles_swe"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        return [e.compiles for e in ctx.edit_history]
