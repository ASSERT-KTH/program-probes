from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCorrectSweProbe(ProbeAdapter):
    """Dynamic probe: each position labeled True if the tests were fully resolved at the last edit."""

    name = "currently_correct_swe"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels = ctx.sample.get("label_sequence_currently_correct")
        if labels is None:
            return [None] * ctx.n_captured_steps
        return labels
