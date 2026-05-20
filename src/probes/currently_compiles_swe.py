from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCompilesSwEProbe(ProbeAdapter):
    """Dynamic probe: each position labeled True if the code compiled at the last edit."""

    name = "currently_compiles_swe"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels = ctx.sample.get("label_sequence_currently_compiles")
        if labels is None:
            return [None] * ctx.n_captured_steps
        return labels
